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
import os

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

