import copy
import gc
import os
from glob import glob
from typing import Literal

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import vision_aided_loss
import wandb
from cleanfid.fid import (build_feature_extractor, frechet_distance,
                          get_folder_features)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from lightning import LightningModule
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchvision import transforms
from tqdm import tqdm

from models.components.fundus_translate import (VAE_decode, VAE_encode,
                                                VAE_encode_oneway, VAE_decode_oneway,
                                                forward_with_networks,
                                                forward_with_networks_oneway,
                                                get_traininable_params,
                                                initialize_unet, initialize_vae,
                                                my_vae_decoder_fwd,
                                                my_vae_encoder_fwd)
from models.components.dino_struct import DinoStructureLoss
from models.components.high_frequency_filter import *


class FundusLitModule(LightningModule):
    """Example of a `LightningModule` for Fundus image translation.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        output_dir: str = None,
        lora_rank_unet: int = 128,
        lora_rank_vae: int = 4,
        allow_tf32: bool = True,
        enable_xformers_memory_efficient_attention: bool = True,
        gradient_checkpointing: bool = False,
        gan_disc_type: str = "vagan_clip",
        gan_loss_type: str = "multilevel_sigmoid",
        lambda_cycle: float = 1.0,
        lambda_cycle_lpips: float = 10.0,
        lambda_gan: float = 0.5,
        lambda_idt: float = 1,
        lambda_idt_lpips: float = 1.0,
        learning_rate: float = 5e-6,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        lr_num_cycles: int = 1,
        lr_power: float = 1.0,
        max_grad_norm: float = 10.0,
        revision: str = None,
        viz_freq: int = 50,
        validation_num_images: int = 10,
    ):
        super().__init__()
        if not output_dir:
            raise ValueError("[DATAMODULE]: output_dir must be provided.")
        self.save_hyperparameters()

        self.automatic_optimization = False  # Manual optimization
        self.n_criterion = 5

        self.noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
        self.noise_scheduler_1step.set_timesteps(1)

        self.unet, self.l_modules_unet_encoder, self.l_modules_unet_decoder, self.l_modules_unet_others = initialize_unet(
            self.hparams.lora_rank_unet, return_lora_module_names=True)

        self.vae_a2b, self.vae_lora_target_modules = initialize_vae(
            self.hparams.lora_rank_vae, return_lora_module_names=True)
        self.vae_b2a = copy.deepcopy(self.vae_a2b)
        self.vae_enc = VAE_encode(self.vae_a2b, vae_b2a=self.vae_b2a)
        self.vae_dec = VAE_decode(self.vae_a2b, vae_b2a=self.vae_b2a)

        if self.hparams.enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        if self.hparams.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if self.hparams.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.unet.conv_in.requires_grad_(True)

        # Discriminators
        if self.hparams.gan_disc_type == "vagan_clip":
            self.net_disc_a = vision_aided_loss.Discriminator(
                cv_type='clip', loss_type=self.hparams.gan_loss_type, device='cuda')
            self.net_disc_a.cv_ensemble.requires_grad_(False)
            self.net_disc_b = vision_aided_loss.Discriminator(
                cv_type='clip', loss_type=self.hparams.gan_loss_type, device='cuda')
            self.net_disc_b.cv_ensemble.requires_grad_(False)

        self.crit_cycle, self.crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()
        self.net_lpips = lpips.LPIPS(net='vgg')
        self.net_lpips.requires_grad_(False)

        self.params_gen = get_traininable_params(
            self.unet, self.vae_a2b, self.vae_b2a)
        self.params_disc = list(self.net_disc_a.parameters()) + \
            list(self.net_disc_b.parameters())

        for name, module in self.net_disc_a.named_modules():
            if "attn" in name:
                module.fused_attn = False
        for name, module in self.net_disc_b.named_modules():
            if "attn" in name:
                module.fused_attn = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feat_model = build_feature_extractor(
            "clean", device, use_dataparallel=False)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, domain: str, text_emb: torch.Tensor) -> torch.Tensor:
        return forward_with_networks(
            x, domain, self.vae_enc, self.unet, self.vae_dec, self.noise_scheduler_1step, timesteps, text_emb
        )

    def training_step(self, batch, batch_idx):
        h = self.hparams
        dm = self.trainer.datamodule
        img_a = batch["pixel_values_src"]
        img_b = batch["pixel_values_tgt"]

        opt_gen, opt_disc = self.optimizers()
        sch_gen, sch_disc = self.lr_schedulers()

        bsz = img_a.shape[0]
        fixed_a2b_emb = dm.fixed_a2b_emb_base.repeat(
            bsz, 1, 1).to(img_a.device)
        fixed_b2a_emb = dm.fixed_b2a_emb_base.repeat(
            bsz, 1, 1).to(img_a.device)
        timesteps = torch.tensor(
            [self.noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

        # Cycle losses
        cyc_fake_b = self.forward(img_a, timesteps, "a2b", fixed_a2b_emb)
        cyc_rec_a = self.forward(cyc_fake_b, timesteps, "b2a", fixed_b2a_emb)
        loss_cycle_a = self.crit_cycle(cyc_rec_a, img_a) * h.lambda_cycle
        loss_cycle_a += self.net_lpips(cyc_rec_a,
                                       img_a).mean() * h.lambda_cycle_lpips

        cyc_fake_a = self.forward(img_b, timesteps, "b2a", fixed_b2a_emb)
        cyc_rec_b = self.forward(cyc_fake_a, timesteps, "a2b", fixed_a2b_emb)
        loss_cycle_b = self.crit_cycle(cyc_rec_b, img_b) * h.lambda_cycle
        loss_cycle_b += self.net_lpips(cyc_rec_b,
                                       img_b).mean() * h.lambda_cycle_lpips

        self.manual_backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
        # import pdb; pdb.set_trace()
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_gen,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_gen.step()
        sch_gen.step()
        opt_gen.zero_grad()

        # Generator GAN losses
        fake_a = self.forward(img_b, timesteps, "b2a", fixed_b2a_emb)
        fake_b = self.forward(img_a, timesteps, "a2b", fixed_a2b_emb)
        loss_gan_a = self.net_disc_a(fake_b, for_G=True).mean() * h.lambda_gan
        loss_gan_b = self.net_disc_b(fake_a, for_G=True).mean() * h.lambda_gan

        self.manual_backward(loss_gan_a + loss_gan_b, retain_graph=False)
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_gen,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_gen.step()
        sch_gen.step()
        opt_gen.zero_grad()
        opt_disc.zero_grad()

        # Identity losses
        idt_a = self.forward(img_b, timesteps, "a2b", fixed_a2b_emb)
        loss_idt_a = self.crit_idt(idt_a, img_b) * h.lambda_idt
        loss_idt_a += self.net_lpips(idt_a, img_b).mean() * h.lambda_idt_lpips
        idt_b = self.forward(img_a, timesteps, "b2a", fixed_b2a_emb)
        loss_idt_b = self.crit_idt(idt_b, img_a) * h.lambda_idt
        loss_idt_b += self.net_lpips(idt_b, img_a).mean() * h.lambda_idt_lpips
        loss_g_idt = loss_idt_a + loss_idt_b

        self.manual_backward(loss_g_idt, retain_graph=False)
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_gen,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_gen.step()
        sch_gen.step()
        opt_gen.zero_grad()

        # Discriminator losses
        loss_D_A_fake = self.net_disc_a(
            fake_b.detach(), for_real=False).mean() * h.lambda_gan
        loss_D_B_fake = self.net_disc_b(
            fake_a.detach(), for_real=False).mean() * h.lambda_gan
        loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5

        self.manual_backward(loss_D_fake, retain_graph=False)
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_disc,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_disc.step()
        sch_disc.step()
        opt_disc.zero_grad()

        loss_D_A_real = self.net_disc_a(
            img_b, for_real=True).mean() * h.lambda_gan
        loss_D_B_real = self.net_disc_b(
            img_a, for_real=True).mean() * h.lambda_gan
        loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5

        self.manual_backward(loss_D_real, retain_graph=False)
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_disc,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_disc.step()
        sch_disc.step()
        opt_disc.zero_grad()

        log_dict = {
            "cycle_a": loss_cycle_a.detach(),
            "cycle_b": loss_cycle_b.detach(),
            "gan_a": loss_gan_a.detach(),
            "gan_b": loss_gan_b.detach(),
            "idt_a": loss_idt_a.detach(),
            "idt_b": loss_idt_b.detach(),
            "disc_a": (loss_D_A_fake + loss_D_A_real).detach(),
            "disc_b": (loss_D_B_fake + loss_D_B_real).detach()
        }

        # log metrics
        self.trainer.logger.experiment.log(
            log_dict,
            step=self.global_step
        )

        if self.trainer.is_global_zero and batch_idx % h.viz_freq == 0:
            img_grid = [wandb.Image(torchvision.utils.make_grid([
                        img_a[idx].float().detach().cpu(),
                        fake_b[idx].float().detach().cpu(),
                        cyc_rec_a[idx].float().detach().cpu(),
                        img_b[idx].float().detach().cpu(),
                        fake_a[idx].float().detach().cpu(),
                        cyc_rec_b[idx].float().detach().cpu()], nrow=3),
                caption=f"idx={idx}") for idx in range(bsz)]

            self.trainer.logger.experiment.log(
                {"train/vis_train": img_grid},
                step=self.global_step
            )

        return loss_cycle_a + loss_cycle_b + loss_cycle_b + loss_gan_a + loss_gan_b + loss_gan_b + loss_g_idt + loss_D_fake + loss_D_real + loss_D_real

    def validation_step(self, batch, batch_idx):
        if batch_idx != 1:
            return
        h = self.hparams
        dm = self.trainer.datamodule
        a2b_ref_mu = np.load(os.path.join(dm.output_dir, "a2b_ref_mu.npy"))
        a2b_ref_sigma = np.load(os.path.join(
            dm.output_dir, "a2b_ref_sigma.npy"))
        b2a_ref_mu = np.load(os.path.join(dm.output_dir, "b2a_ref_mu.npy"))
        b2a_ref_sigma = np.load(os.path.join(
            dm.output_dir, "b2a_ref_sigma.npy"))

        # Visualize dataset
        l_images_src_test = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            l_images_src_test.extend(
                glob(os.path.join(dm.data_dir, "test_A", ext)))
        l_images_tgt_test = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            l_images_tgt_test.extend(
                glob(os.path.join(dm.data_dir, "test_B", ext)))
        l_images_src_test, l_images_tgt_test = sorted(
            l_images_src_test), sorted(l_images_tgt_test)

        img_a = batch["pixel_values_src"]
        img_b = batch["pixel_values_tgt"]

        bsz = img_a.shape[0]
        fixed_a2b_emb = dm.fixed_a2b_emb_base.repeat(
            bsz, 1, 1).to(img_a.device).to(img_a.device)
        fixed_b2a_emb = dm.fixed_b2a_emb_base.repeat(
            bsz, 1, 1).to(img_a.device).to(img_a.device)
        timesteps = torch.tensor(
            [self.noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

        with torch.no_grad():
            fake_b = self.forward(img_a, timesteps, "a2b", fixed_a2b_emb)
            fake_a = self.forward(img_b, timesteps, "b2a", fixed_b2a_emb)

        lpips_a2b = self.net_lpips(fake_b, img_b).mean()
        lpips_b2a = self.net_lpips(fake_a, img_a).mean()

        if self.trainer.is_global_zero:
            eval_unet = self.unet.eval()
            eval_vae_enc = self.vae_enc.eval()
            eval_vae_dec = self.vae_dec.eval()
            # if no directory, then make one

            if not os.path.exists(os.path.join(h.output_dir, "checkpoints")):
                os.makedirs(os.path.join(h.output_dir, "checkpoints"))
            outf = os.path.join(h.output_dir, "checkpoints",
                                f"model_{self.global_step}.pkl")
            sd = {}
            sd["l_target_modules_encoder"] = self.l_modules_unet_encoder
            sd["l_target_modules_decoder"] = self.l_modules_unet_decoder
            sd["l_modules_others"] = self.l_modules_unet_others
            sd["rank_unet"] = h.lora_rank_unet
            sd["sd_encoder"] = get_peft_model_state_dict(
                eval_unet, adapter_name="default_encoder")
            sd["sd_decoder"] = get_peft_model_state_dict(
                eval_unet, adapter_name="default_decoder")
            sd["sd_other"] = get_peft_model_state_dict(
                eval_unet, adapter_name="default_others")
            sd["rank_vae"] = h.lora_rank_vae
            sd["vae_lora_target_modules"] = self.vae_lora_target_modules
            sd["sd_vae_enc"] = eval_vae_enc.state_dict()
            sd["sd_vae_dec"] = eval_vae_dec.state_dict()
            torch.save(sd, outf)
            gc.collect()
            torch.cuda.empty_cache()

            _timesteps = torch.tensor(
                [self.noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda").long()
            net_dino = DinoStructureLoss()
            """
            Evaluate "A->B"
            """
            fid_output_dir = os.path.join(
                h.output_dir, f"fid-{self.global_step}/samples_a2b")
            os.makedirs(fid_output_dir, exist_ok=True)
            l_dino_scores_a2b = []
            # get val input images from domain a
            for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                if idx > h.validation_num_images and h.validation_num_images > 0:
                    break
                outf = os.path.join(fid_output_dir, f"{idx}.png")
                with torch.no_grad():
                    input_img = Image.open(input_img_path).convert("RGB")
                    img_a = transforms.ToTensor()(input_img)
                    img_a = transforms.Normalize([0.5], [0.5])(
                        img_a).unsqueeze(0).cuda()
                    eval_fake_b = forward_with_networks(img_a, "a2b", eval_vae_enc, eval_unet,
                                                        eval_vae_dec, self.noise_scheduler_1step, _timesteps, fixed_a2b_emb[0:1])
                    eval_fake_b_pil = transforms.ToPILImage()(
                        eval_fake_b[0] * 0.5 + 0.5)
                    eval_fake_b_pil.save(outf)
                    a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                    b = net_dino.preprocess(
                        eval_fake_b_pil).unsqueeze(0).cuda()
                    dino_ssim = net_dino.calculate_global_ssim_loss(
                        a, b).item()
                    l_dino_scores_a2b.append(dino_ssim)
            dino_score_a2b = np.mean(l_dino_scores_a2b)
            gen_features = get_folder_features(fid_output_dir, model=self.feat_model, num_workers=0, num=None,
                                               shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                               mode="clean", custom_fn_resize=None, description="", verbose=True,
                                               custom_image_tranform=None)
            ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(
                gen_features, rowvar=False)
            score_fid_a2b = frechet_distance(
                a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
            print(
                f"step={self.global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")

            """
            compute FID for "B->A"
            """
            fid_output_dir = os.path.join(
                h.output_dir, f"fid-{self.global_step}/samples_b2a")
            os.makedirs(fid_output_dir, exist_ok=True)
            l_dino_scores_b2a = []
            # get val input images from domain b
            for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
                if idx > h.validation_num_images and h.validation_num_images > 0:
                    break
                outf = os.path.join(fid_output_dir, f"{idx}.png")
                with torch.no_grad():
                    input_img = Image.open(input_img_path).convert("RGB")
                    img_b = transforms.ToTensor()(input_img)
                    img_b = transforms.Normalize([0.5], [0.5])(
                        img_b).unsqueeze(0).cuda()
                    eval_fake_a = forward_with_networks(img_b, "b2a", eval_vae_enc, eval_unet,
                                                        eval_vae_dec, self.noise_scheduler_1step, _timesteps, fixed_b2a_emb[0:1])
                    eval_fake_a_pil = transforms.ToPILImage()(
                        eval_fake_a[0] * 0.5 + 0.5)
                    eval_fake_a_pil.save(outf)
                    a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                    b = net_dino.preprocess(
                        eval_fake_a_pil).unsqueeze(0).cuda()
                    dino_ssim = net_dino.calculate_global_ssim_loss(
                        a, b).item()
                    l_dino_scores_b2a.append(dino_ssim)
            dino_score_b2a = np.mean(l_dino_scores_b2a)
            gen_features = get_folder_features(fid_output_dir, model=self.feat_model, num_workers=0, num=None,
                                               shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                               mode="clean", custom_fn_resize=None, description="", verbose=True,
                                               custom_image_tranform=None)
            ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(
                gen_features, rowvar=False)
            score_fid_b2a = frechet_distance(
                b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
            print(
                f"step={self.global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}")
            log_dict = {
                "val/lpips_a2b": lpips_a2b,
                "val/lpips_b2a": lpips_b2a,
                "val/fid_a2b": score_fid_a2b,
                "val/fid_b2a": score_fid_b2a,
                "val/dino_struct_a2b": dino_score_a2b,
                "val/dino_struct_b2a": dino_score_b2a
            }
            self.trainer.logger.experiment.log(
                log_dict,
                step=self.global_step
            )
            del net_dino  # free up memory

            self.unet.train()
            self.vae_enc.train()
            self.vae_dec.train()

    def configure_optimizers(self):
        h = self.hparams
        params_gen = get_traininable_params(
            self.unet, self.vae_a2b, self.vae_b2a)
        optimizer_gen = torch.optim.AdamW(
            params_gen, lr=h.learning_rate, betas=(h.adam_beta1, h.adam_beta2),
            weight_decay=h.adam_weight_decay, eps=h.adam_epsilon
        )

        params_disc = list(self.net_disc_a.parameters()) + \
            list(self.net_disc_b.parameters())
        optimizer_disc = torch.optim.AdamW(
            params_disc, lr=h.learning_rate, betas=(
                h.adam_beta1, h.adam_beta2),
            weight_decay=h.adam_weight_decay, eps=h.adam_epsilon
        )

        total_steps = self.trainer.max_steps
        lr_scheduler_gen = get_scheduler(
            h.lr_scheduler, optimizer=optimizer_gen,
            num_warmup_steps=h.lr_warmup_steps,
            num_training_steps=total_steps, num_cycles=h.lr_num_cycles, power=h.lr_power
        )

        lr_scheduler_disc = get_scheduler(
            h.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=h.lr_warmup_steps,
            num_training_steps=total_steps, num_cycles=h.lr_num_cycles, power=h.lr_power
        )

        return [
            {"optimizer": optimizer_gen, "lr_scheduler": {
                "scheduler": lr_scheduler_gen, "interval": "step"}},
            {"optimizer": optimizer_disc, "lr_scheduler": {
                "scheduler": lr_scheduler_disc, "interval": "step"}}
        ]

    @property
    def global_step(self) -> int:
        return int(super().global_step // self.n_criterion)


class FundusOneWayLitModule(LightningModule):
    """Example of a `LightningModule` for Fundus image translation.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        output_dir: str = None,
        lora_rank_unet: int = 128,
        lora_rank_vae: int = 4,
        allow_tf32: bool = True,
        enable_xformers_memory_efficient_attention: bool = True,
        gradient_checkpointing: bool = False,
        gan_disc_type: str = "vagan_clip",
        gan_loss_type: str = "multilevel_sigmoid",
        lambda_l2: float = 1.0,
        lambda_lhfc: float = 1.0,
        lambda_lpips: float = 5.0,
        lambda_gan: float = 0.5,  # CAN ADJUST
        lambda_idt: float = 1.0,
        lambda_idt_lpips: float = 1.0,
        learning_rate: float = 5e-6,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        lr_num_cycles: int = 1,
        lr_power: float = 1.0,
        max_grad_norm: float = 10.0,
        revision: str = None,
        viz_freq: int = 50,
        validation_num_images: int = 10,
        ckpt_direction: Literal["a2b", "b2a"] = "b2a",
        pretrained_path: str = None
    ):
        super().__init__()
        if not output_dir:
            raise ValueError("[DATAMODULE]: output_dir must be provided.")
        self.save_hyperparameters()

        self.automatic_optimization = False  # Manual optimization
        self.n_criterion = 2

        self.noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
        self.noise_scheduler_1step.set_timesteps(1)

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-turbo", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="unet")
        self.vae.requires_grad_(False)
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(
            self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(
            self.vae.decoder, self.vae.decoder.__class__)
        self.vae.requires_grad_(True)
        self.vae.train()
        # add the skip connection convs
        self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.ignore_skip = False

        self.unet.requires_grad_(False)
        self.unet.train()
        if pretrained_path is not None:
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd, ckpt_direction)
        else:
            self.unet, self.l_modules_unet_encoder, self.l_modules_unet_decoder, self.l_modules_unet_others = initialize_unet(
                self.hparams.lora_rank_unet, return_lora_module_names=True)
            self.vae, self.vae_lora_target_modules = initialize_vae(
                self.hparams.lora_rank_vae, return_lora_module_names=True)
            self.vae_enc = VAE_encode_oneway(self.vae)
            self.vae_dec = VAE_decode_oneway(self.vae)
        self.unet.train()
        # return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

        if self.hparams.enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        if self.hparams.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if self.hparams.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.unet.conv_in.requires_grad_(True)

        # Discriminators
        if self.hparams.gan_disc_type == "vagan_clip":
            self.net_disc = vision_aided_loss.Discriminator(
                cv_type='clip', loss_type=self.hparams.gan_loss_type, device='cuda')
            self.net_disc.cv_ensemble.requires_grad_(False)

        self.crit_idt = torch.nn.L1Loss()
        self.net_lpips = lpips.LPIPS(net='vgg')
        self.net_lpips.requires_grad_(False)

        self.params_gen = get_traininable_params(self.unet, self.vae)
        self.hfc_filter = HFCFilter()
        self.crit_hfc = torch.nn.L1Loss()
        # self.params_disc = list(self.net_disc.parameters())

        # for name, module in self.net_disc.named_modules():
        #     if "attn" in name:
        #         module.fused_attn = False

        self.psnr = psnr()
        self.ssim = ssim()
        self.test_psnr_list = []
        self.test_ssim_list = []

    def load_ckpt_from_state_dict(self, sd, direction: Literal["a2b", "b2a"]):
        """
        load vae of direction from state_dict
        """
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(
            lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(
            lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])
        self.unet.set_adapter(
            ["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(
            r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        if direction == "a2b":
            one_way_enc = {
                k: v for k, v in sd["sd_vae_enc"].items() if k.startswith("vae.")}
            one_way_dec = {
                k: v for k, v in sd["sd_vae_dec"].items() if k.startswith("vae.")}
        elif direction == "b2a":
            one_way_enc = {k.replace("vae_b2a.", "vae."): v for k, v in sd["sd_vae_enc"].items(
            ) if k.startswith("vae_b2a.")}
            one_way_dec = {k.replace("vae_b2a.", "vae."): v for k, v in sd["sd_vae_dec"].items(
            ) if k.startswith("vae_b2a.")}

        self.vae_enc = VAE_encode_oneway(self.vae)
        self.vae_enc.load_state_dict(one_way_enc)
        self.vae_dec = VAE_decode_oneway(self.vae)
        self.vae_dec.load_state_dict(one_way_dec)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        return forward_with_networks_oneway(
            x, self.vae_enc, self.unet, self.vae_dec, self.noise_scheduler_1step, timesteps, text_emb
        )

    def training_step(self, batch, batch_idx):
        h = self.hparams
        dm = self.trainer.datamodule
        x_src = batch["pixel_values_src"]
        x_tgt = batch["pixel_values_tgt"]
        x_mask = batch["pixel_values_mask"]

        opt_gen = self.optimizers()
        sch_gen = self.lr_schedulers()

        bsz = x_src.shape[0]
        fixed_emb = dm.fixed_emb_base.repeat(
            bsz, 1, 1).to(x_src.device)
        timesteps = torch.tensor(
            [self.noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=x_src.device).long()

        # Cycle losses
        x_tgt_pred = self.forward(x_src, timesteps, fixed_emb)
        loss_l2 = F.mse_loss(x_tgt_pred, x_tgt, reduction="mean") * h.lambda_l2
        loss_lpips = self.net_lpips(x_tgt_pred, x_tgt).mean() * h.lambda_lpips
        x_tgt_pred_hfc = hfc_mul_mask(self.hfc_filter, x_tgt_pred, x_mask)
        x_tgt_hfc = hfc_mul_mask(self.hfc_filter, x_tgt, x_mask)
        loss_hfc = self.crit_hfc(x_tgt_hfc, x_tgt_pred_hfc) * h.lambda_lhfc
        loss = loss_l2 + loss_lpips + loss_hfc
        self.manual_backward(loss, retain_graph=False)
        # import pdb; pdb.set_trace()
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_gen,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_gen.step()
        sch_gen.step()
        opt_gen.zero_grad()

        # # Fool the Discriminator
        # if h.lambda_gan > 0:
        #     x_tgt_pred = self.forward(x_src, timesteps, fixed_emb)
        #     loss_gan = self.net_disc(x_tgt_pred, for_G=True).mean() * h.lambda_gan

        # self.manual_backward(loss_gan, retain_graph=False)
        # if self.trainer.strategy.accelerator.auto_device_count() > 0:
        #     self.clip_gradients(
        #         optimizer=opt_gen,
        #         gradient_clip_val=h.max_grad_norm,
        #         gradient_clip_algorithm="norm"
        #     )

        # opt_gen.step()
        # sch_gen.step()
        # opt_gen.zero_grad()

        # Identity losses
        x_tgt_idt = self.forward(x_tgt, timesteps, fixed_emb)
        loss_g_idt = self.crit_idt(x_tgt_idt, x_tgt) * h.lambda_idt
        loss_g_idt_lpips = self.net_lpips(
            x_tgt_idt, x_tgt).mean() * h.lambda_idt_lpips
        loss_g = loss_g_idt + loss_g_idt_lpips

        self.manual_backward(loss_g, retain_graph=False)
        if self.trainer.strategy.accelerator.auto_device_count() > 0:
            self.clip_gradients(
                optimizer=opt_gen,
                gradient_clip_val=h.max_grad_norm,
                gradient_clip_algorithm="norm"
            )

        opt_gen.step()
        sch_gen.step()
        opt_gen.zero_grad()
        # # Fake image
        # loss_D_fake = self.net_disc(x_tgt_pred.detach(), for_real=False).mean() * h.lambda_gan

        # self.manual_backward(loss_D_fake, retain_graph=False)
        # if self.trainer.strategy.accelerator.auto_device_count() > 0:
        #     self.clip_gradients(
        #         optimizer=opt_disc,
        #         gradient_clip_val=h.max_grad_norm,
        #         gradient_clip_algorithm="norm"
        #     )

        # opt_disc.step()
        # sch_disc.step()
        # opt_disc.zero_grad()

        # # Real image
        # loss_D_real = self.net_disc(x_tgt.detach(), for_real=True).mean() * h.lambda_gan

        # self.manual_backward(loss_D_real, retain_graph=False)
        # if self.trainer.strategy.accelerator.auto_device_count() > 0:
        #     self.clip_gradients(
        #         optimizer=opt_disc,
        #         gradient_clip_val=h.max_grad_norm,
        #         gradient_clip_algorithm="norm"
        #     )

        # opt_disc.step()
        # sch_disc.step()
        # opt_disc.zero_grad()

        log_dict = {
            "l2": loss_l2.detach(),
            "lpips": loss_lpips.detach(),
            "identity": loss_g.detach()
            # "D_fake": loss_D_fake.detach(),
            # "D_real": loss_D_real.detach()
        }

        # log metrics
        self.trainer.logger.experiment.log(
            log_dict,
            step=self.global_step
        )

        if self.trainer.is_global_zero and self.global_step % h.viz_freq == 0:
            img_grid = [wandb.Image(torchvision.utils.make_grid([
                        x_src[idx].float().detach().cpu(),
                        x_tgt_pred[idx].float().detach().cpu(),
                        x_tgt[idx].float().detach().cpu()], nrow=3),
                caption=f"idx={idx}") for idx in range(bsz)]

            self.trainer.logger.experiment.log(
                {"train/vis_train": img_grid}
            )

        return loss + loss_g

    def validation_step(self, batch, batch_idx):
        if batch_idx == 20:
            self.log("val/psnr_avg", np.average(np.array(self.test_psnr_list)),
                     on_step=True, prog_bar=True)
            self.log("val/ssim_avg", np.average(np.array(self.test_ssim_list)),
                     on_step=True, prog_bar=True)
            self.test_psnr_list = []
            self.test_ssim_list = []
            return
        if batch_idx > 20:
            return
        h = self.hparams
        dm = self.trainer.datamodule
        x_src = batch["pixel_values_src"]
        x_tgt = batch["pixel_values_tgt"]

        bsz = x_src.shape[0]
        fixed_emb = dm.fixed_emb_base.repeat(
            bsz, 1, 1).to(x_src.device)
        timesteps = torch.tensor(
            [self.noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=x_src.device).long()

        # Cycle losses
        with torch.no_grad():
            x_tgt_pred = self.forward(x_src, timesteps, fixed_emb)
            self.test_psnr_list.append(self.psnr(x_tgt, x_tgt_pred).cpu())
            self.test_ssim_list.append(self.ssim(x_tgt, x_tgt_pred).cpu())

    def test_step(self, batch, batch_idx):
        h = self.hparams
        dm = self.trainer.datamodule
        x_src = batch["pixel_values_src"]
        x_tgt = batch["pixel_values_tgt"]

        bsz = x_src.shape[0]
        fixed_emb = dm.fixed_emb_base.repeat(
            bsz, 1, 1).to(x_src.device)
        timesteps = torch.tensor(
            [self.noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=x_src.device).long()

        # Cycle losses
        with torch.no_grad():
            x_tgt_pred = self.forward(x_src, timesteps, fixed_emb)
            self.test_psnr_list.append(self.psnr(x_tgt, x_tgt_pred).cpu())
            self.test_ssim_list.append(self.ssim(x_tgt, x_tgt_pred).cpu())

        x_tgt_pred = x_tgt_pred * 0.5 + 0.5
        torchvision.utils.save_image(x_tgt_pred, f'/public/home/wangzh1/miccai2025/FR-UNet/data/enhanced_ours/{batch_idx:06}.jpeg')
        
        self.log("test/psnr_avg", np.average(np.array(self.test_psnr_list)),
                 on_step=True, prog_bar=True)
        self.log("test/ssim_avg", np.average(np.array(self.test_ssim_list)),
                 on_step=True, prog_bar=True)

    def configure_optimizers(self):
        h = self.hparams
        params_gen = get_traininable_params(self.unet, self.vae)
        optimizer_gen = torch.optim.AdamW(
            params_gen, lr=h.learning_rate, betas=(h.adam_beta1, h.adam_beta2),
            weight_decay=h.adam_weight_decay, eps=h.adam_epsilon
        )

        # params_disc = list(self.net_disc.parameters())
        # optimizer_disc = torch.optim.AdamW(
        #     params_disc, lr=h.learning_rate, betas=(
        #         h.adam_beta1, h.adam_beta2),
        #     weight_decay=h.adam_weight_decay, eps=h.adam_epsilon
        # )

        total_steps = self.trainer.max_steps
        lr_scheduler_gen = get_scheduler(
            h.lr_scheduler, optimizer=optimizer_gen,
            num_warmup_steps=h.lr_warmup_steps,
            num_training_steps=total_steps, num_cycles=h.lr_num_cycles, power=h.lr_power
        )

        # lr_scheduler_disc = get_scheduler(
        #     h.lr_scheduler, optimizer=optimizer_disc,
        #     num_warmup_steps=h.lr_warmup_steps,
        #     num_training_steps=total_steps, num_cycles=h.lr_num_cycles, power=h.lr_power
        # )

        return [
            {"optimizer": optimizer_gen, "lr_scheduler": {
                "scheduler": lr_scheduler_gen, "interval": "step"}}
            # {"optimizer": optimizer_disc, "lr_scheduler": {
            #     "scheduler": lr_scheduler_disc, "interval": "step"}}
        ]
