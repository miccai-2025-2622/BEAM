import glob
import os
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from cleanfid.fid import (build_feature_extractor, frechet_distance,
                          get_folder_features)
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from data.components.fundus_dataset import (PairedFundusDataset,
                                            UnpairedFundusDataset,
                                            build_transform)


class FundusDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = None,
        train_image_prep: str = "no_resize",
        val_image_prep: str = "no_resize",
        batch_size: int = 1,
        num_workers: int = 0
    ) -> None:
        super().__init__()

        if not data_dir or not output_dir:
            raise ValueError("[DATAMODULE]: data_dir and output_dir must be provided.")
        
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data_train: Optional[UnpairedFundusDataset] = None
        self.data_val: Optional[UnpairedFundusDataset] = None
        self.data_test: Optional[UnpairedFundusDataset] = None

        self.fixed_caption_src = None
        self.fixed_caption_tgt = None
        self.fixed_a2b_emb_base = None
        self.fixed_b2a_emb_base = None

    def prepare_data(self) -> None:
        h = self.hparams
        # This is run on a single process.
        # Download tokenizer and CLIP if not done before
        AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False)
        CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")

        l_images_src_test = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            l_images_src_test.extend(glob.glob(os.path.join(h.data_dir, "test_A", ext)))
        l_images_tgt_test = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            l_images_tgt_test.extend(glob.glob(os.path.join(h.data_dir, "test_B", ext)))
        l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        feat_model = build_feature_extractor("clean", device, use_dataparallel=False) 

        T_val = build_transform(h.val_image_prep)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(h.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device(device),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
        """
        FID reference statistics for B -> A translation
        """
        # transform all images according to the validation transform and save them
        output_dir_ref = os.path.join(h.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device(device),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

        np.save(os.path.join(h.output_dir, "a2b_ref_mu.npy"), a2b_ref_mu)
        np.save(os.path.join(h.output_dir, "a2b_ref_sigma.npy"), a2b_ref_sigma)
        np.save(os.path.join(h.output_dir, "b2a_ref_mu.npy"), b2a_ref_mu)
        np.save(os.path.join(h.output_dir, "b2a_ref_sigma.npy"), b2a_ref_sigma)

    def setup(self, stage: Optional[str] = None) -> None:
        # Called on every GPU. Load datasets and text encoder here.

        tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False)
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
        text_encoder.requires_grad_(False) 

        self.data_train = UnpairedFundusDataset(
            dataset_folder=self.hparams.data_dir,
            split="train",
            image_prep=self.hparams.train_image_prep,
            tokenizer=tokenizer,
        )

        self.data_val = UnpairedFundusDataset(
            dataset_folder=self.hparams.data_dir,
            split="test",
            image_prep=self.hparams.val_image_prep,
            tokenizer=tokenizer,
        )

        self.data_test = UnpairedFundusDataset(
            dataset_folder=self.hparams.data_dir,
            split="test",
            image_prep=self.hparams.val_image_prep,
            tokenizer=tokenizer,
        )

        # Extract fixed captions from the training dataset (it stores them)
        self.fixed_caption_src = self.data_train.fixed_caption_src
        self.fixed_caption_tgt = self.data_train.fixed_caption_tgt

        with torch.no_grad():
            self.fixed_a2b_emb_base = text_encoder(self.data_train.fixed_a2b_tokens.unsqueeze(0))[0].detach()
            self.fixed_b2a_emb_base = text_encoder(self.data_train.fixed_b2a_tokens.unsqueeze(0))[0].detach()

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )


class FundusOneWayDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = None,
        train_image_prep: str = "resize_286_randomcrop_256x256_hflip",
        val_image_prep: str = "no_resize",
        batch_size: int = 1,
        num_workers: int = 8,
        type: Literal["TMI", "MY", "SEG"] = "MY"
    ) -> None:
        super().__init__()

        if not data_dir or not output_dir:
            raise ValueError("[DATAMODULE]: data_dir and output_dir must be provided.")
        
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data_train: Optional[PairedFundusDataset] = None
        self.data_val: Optional[PairedFundusDataset] = None
        self.data_test: Optional[PairedFundusDataset] = None

        self.fixed_caption_src = None
        self.fixed_caption_tgt = None
        self.fixed_a2b_emb_base = None
        self.fixed_b2a_emb_base = None

    def prepare_data(self) -> None:
        # This is run on a single process.
        # Download tokenizer and CLIP if not done before
        AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False)
        CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")


    def setup(self, stage: Optional[str] = None) -> None:
        # Called on every GPU. Load datasets and text encoder here.
        h = self.hparams
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False)
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
        text_encoder.requires_grad_(False) 
        if stage == 'fit':
            self.data_train = PairedFundusDataset(
                dataset_folder=h.data_dir,
                split="train",
                image_prep=h.train_image_prep,
                tokenizer=tokenizer,
                type=h.type
            )

            self.data_val = PairedFundusDataset(
                dataset_folder=h.data_dir,
                split="test",
                image_prep=h.val_image_prep,
                tokenizer=tokenizer,
                type=h.type
            )
            self.fixed_caption_tgt = self.data_train.fixed_caption_tgt
            with torch.no_grad():
                self.fixed_emb_base = text_encoder(self.data_train.fixed_tokens.unsqueeze(0))[0].detach()
        if stage == 'test':
            self.data_test = PairedFundusDataset(
                dataset_folder=h.data_dir,
                split="test",
                image_prep=h.val_image_prep,
                tokenizer=tokenizer,
                type=h.type
            )
            self.fixed_caption_tgt = self.data_test.fixed_caption_tgt
            with torch.no_grad():
                self.fixed_emb_base = text_encoder(self.data_test.fixed_tokens.unsqueeze(0))[0].detach()


    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

if __name__ == "__main__":
    data_dir = "/storage/data/wangzh1/EyeQ_data_paired"
    output_dir = "/public/home/wangzh1/lightning-hydra-template/logs/lightning_v0.1/runs/2024-12-15_00-29-13"
    train_image_prep = "resize_286_randomcrop_256x256_hflip"
    val_image_prep = "no_resize"
    batch_size = 1
    num_workers = 8
    a = FundusOneWayDataModule(data_dir=data_dir,
                         output_dir=output_dir,
                         train_image_prep=train_image_prep,
                         val_image_prep=val_image_prep,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         type="MY")
    a.prepare_data()
    a.setup(stage='test')
    train_loader = a.train_dataloader()
    batch = next(iter(train_loader))
    import pdb; pdb.set_trace()
    print(f"Number of training batches: {len(train_loader)}")

