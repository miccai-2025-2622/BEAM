import glob
import os
import random
from typing import Literal

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(
                512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class UnpairedFundusDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.fixed_a2b_tokens = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.fixed_b2a_tokens = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(
                glob.glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(
                glob.glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt
        }


class PairedFundusDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_folder,
                 split,
                 image_prep,
                 tokenizer,
                 type: Literal["TMI", "MY", "SEG", "DRIVE"]):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        self.type = type
        if split == "train":
            self.mask_folder = os.path.join(dataset_folder, "train", "mask")
            if self.type == "TMI":
                self.source_folder = os.path.join(
                    dataset_folder, "train", "degraded_tmi")
                self.target_folder = os.path.join(
                    dataset_folder, "train", "gt_tmi")
            elif self.type == "MY":
                self.source_folder = os.path.join(
                    dataset_folder, "train", "degraded")
                self.target_folder = os.path.join(
                    dataset_folder, "train", "gt")
        else:
            self.mask_folder = os.path.join(dataset_folder, "test", "mask")
            if self.type == "TMI":
                self.source_folder = os.path.join(
                    dataset_folder, "test", "degraded_tmi")
                self.target_folder = os.path.join(
                    dataset_folder, "test", "gt_tmi")
            elif self.type == "MY":
                self.source_folder = os.path.join(
                    dataset_folder, "test", "degraded")
                self.target_folder = os.path.join(dataset_folder, "test", "gt")
            elif self.type == "SEG":
                self.source_folder = '/public/home/wangzh1/miccai2025/FR-UNet/data/degraded'
                self.target_folder = '/public/home/wangzh1/miccai2025/FR-UNet/data/original'
                self.mask_folder = '/public/home/wangzh1/miccai2025/FR-UNet/data/mask'

        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "prompt_high_quality.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.fixed_tokens = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]
        # import pdb; pdb.set_trace()
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(
                glob.glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(
                glob.glob(os.path.join(self.target_folder, ext)))
        self.l_imgs_mask = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_mask.extend(
                glob.glob(os.path.join(self.mask_folder, ext)))

        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        if self.type == "MY":
            assert len(self.l_imgs_src) == len(self.l_imgs_tgt)
        return len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        tmi_pattern = ["_001", "_010", "_011", "_100", "_101", "_110", "_111"]
        if self.type == "MY":
            src_tail = ".png"
            tgt_tail = ".png"
        elif self.type == "TMI":
            src_tail = tmi_pattern[random.randint(0, 6)] + ".jpeg"
            tgt_tail = ".jpeg"
        elif self.type == "SEG":
            src_tail = ".jpeg"
            tgt_tail = ".jpeg"

        img_path_src = os.path.join(
            self.source_folder, f"{index:06}" + src_tail)
        img_path_tgt = os.path.join(
            self.target_folder, f"{index:06}" + tgt_tail)
        img_path_mask = os.path.join(
            self.mask_folder, f"{index:06}.jpeg")
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_pil_mask = Image.open(img_path_mask).convert("L")

        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_mask = F.to_tensor(self.T(img_pil_mask))

        target_mask = img_t_tgt == 0
        img_t_src[target_mask] = 0
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "pixel_values_mask": img_t_mask
        }


if __name__ == '__main__':
    import numpy as np
    from transformers import AutoTokenizer, CLIPTextModel
    from torch.utils.data import DataLoader
    from torchmetrics.image import PeakSignalNoiseRatio as psnr
    from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
    test_psnr = psnr().to('cuda')
    test_ssim = ssim().to('cuda')
    psnr_list = []
    ssim_list = []
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False)
    data_test = PairedFundusDataset(
            dataset_folder="/storage/data/wangzh1/EyeQ_data_paired/",
            split="test",
            image_prep="resize_512",
            tokenizer=tokenizer,
            type="TMI"
        )
    data_loader = DataLoader(
            dataset=data_test,
            batch_size=32,
            num_workers=8,
            shuffle=False,
        )
    from tqdm import tqdm 
    for batch in tqdm(data_loader):
        img_t_src = batch["pixel_values_src"].cuda()
        img_t_tgt = batch["pixel_values_tgt"].cuda()
        p = test_psnr(img_t_src, img_t_tgt).cpu()
        s = test_ssim(img_t_src, img_t_tgt).cpu()
        psnr_list.append(p)
        ssim_list.append(s)
        print(p, s)

    print(np.average(np.array(psnr_list)), np.average(np.array(ssim_list)))


