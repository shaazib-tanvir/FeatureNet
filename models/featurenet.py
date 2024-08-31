from models.bagnet import bagnet33
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

class FeatureNet(nn.Module):
    def __init__(self, num_classes, patch_size):
        super().__init__()

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.bagnet = bagnet33(pretrained=True, num_classes=self.num_classes)

    def forward(self, image):
        prediction, features = self.bagnet(image)

        return prediction
