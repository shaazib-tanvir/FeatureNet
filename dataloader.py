# encoding: utf-8

"""
Read images and corresponding labels.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
from PIL import Image, ImageFilter, ImageFile
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip, RandomApply, GaussianNoise, GaussianBlur
import os
import matplotlib.pyplot as plt

conditions = [
    "normal",
    "atelectasis",
    "cardiomegaly",
    "infiltration",
    "nodule",
    "emphysema",
    "pleural thickening",
    "calcified granuloma",
    "opacity",
    "lung/hypoinflation",
    "thoracic vertebrae/degenerative",
    "spine/degenerative",
    "lung/hyperdistention",
    "daphragmatic eventration",
    "calcinosis"
]

num_cls = len(conditions)
conditions_table = {condition: i for i, condition in enumerate(conditions)}


class IUXRayDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 image_list_file: str,
                 to_blur: bool = False,
                 sigma: float = 0,
                 transform=None) -> None:
        self.image_dir = image_dir
        self.data = pd.read_csv(image_list_file)
        self.to_blur = to_blur
        self.sigma = sigma
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(str(os.path.join(self.image_dir, self.data.iloc[index]["filename"]))).convert("RGB")
        if self.to_blur:
            image = image.filter(ImageFilter.GaussianBlur(self.sigma))

        label = conditions_table[str(self.data.iloc[index]["condition"])]

        if self.transform is not None:
            image = self.transform(image)

        label = torch.as_tensor(label, dtype=torch.int64)
        report = self.data.iloc[index]["findings"]
        return image, label, report

    def __len__(self):
        return len(self.data)


class CDD_CESMDataset(Dataset):
    idx2class = ["Normal", "Benign", "Malignant"]
    class2idx = {"Normal": 0, "Benign": 1, "Malignant": 2}

    def __init__(self,
                 image_list_file: str,
                 to_blur: bool = False,
                 sigma: float = 0,
                 transform=None, augment=True) -> None:
        self.data = pd.read_csv(image_list_file)
        self.to_blur = to_blur
        self.sigma = sigma
        self.transform = transform
        self.random_horizontal = RandomHorizontalFlip()
        self.random_vertical = RandomVerticalFlip()
        self.noise = RandomApply(torch.nn.ModuleList([
            GaussianNoise(sigma=0.025)
        ]))
        self.blur = RandomApply(torch.nn.ModuleList([
            GaussianBlur(3, (0.1, 0.5))
        ]))
        self.augment = augment


    def __getitem__(self, index):
        image_path = self.data.iloc[index]["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.to_blur:
            image = image.filter(ImageFilter.GaussianBlur(self.sigma))

        label = self.class2idx[str(self.data.iloc[index]["classification"])]

        if self.transform is not None:
            image = self.transform(image)

        if self.augment:
            image = self.random_horizontal(image)
            image = self.random_vertical(image)
            image = self.noise(image)
            image = self.blur(image)

        label = torch.as_tensor(label, dtype=torch.int64)
        report = self.data.iloc[index]["report"]
        return image, label, report

    def __len__(self):
        return len(self.data)
