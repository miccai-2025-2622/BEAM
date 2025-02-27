import os
import sys
import copy
import torch

import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


class VAE_encode_oneway(nn.Module):
    def __init__(self, vae):
        super(VAE_encode_oneway, self).__init__()
        self.vae = vae

    def forward(self, x):
        _vae = self.vae
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode_oneway(nn.Module):
    def __init__(self, vae):
        super(VAE_decode_oneway, self).__init__()
        self.vae = vae

    def forward(self, x):
        _vae = self.vae
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded

class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
    B = x.shape[0]
    assert direction in ["a2b", "b2a"]
    x_enc = vae_enc(x, direction=direction).to(x.dtype)
    # import pdb; pdb.set_trace()
    model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
    x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
    x_out_decoded = vae_dec(x_out, direction=direction)
    return x_out_decoded

def forward_with_networks_oneway(x, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
    B = x.shape[0]
    x_enc = vae_enc(x).to(x.dtype)
    # import pdb; pdb.set_trace()
    model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
    x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
    x_out_decoded = vae_dec(x_out)
    return x_out_decoded

def get_traininable_params(unet, vae_a2b, vae_b2a=None):
    # add all unet parameters
    params_gen = list(unet.conv_in.parameters())
    unet.conv_in.requires_grad_(True)
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    for n,p in unet.named_parameters():
        if "lora" in n and "default" in n:
            assert p.requires_grad
            params_gen.append(p)
    
    # add all vae_a2b parameters
    for n,p in vae_a2b.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert p.requires_grad
            params_gen.append(p)
    params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
    params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
    params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
    params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

    if vae_b2a is not None:
        # add all vae_b2a parameters
        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
    return params_gen

def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).requires_grad_(True) 
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae
