import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .config import *


class BraTSNet50(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

        oldConv = self.resnet.conv1
        newConv = nn.Conv2d(
            in_channels=4, 
            out_channels=oldConv.out_channels, 
            kernel_size=oldConv.kernel_size, 
            stride=oldConv.stride, 
            padding=oldConv.padding, 
            bias=oldConv.bias is not None
        )

        with torch.no_grad():
            meanWeight = oldConv.weight.mean(dim=1, keepdim=True)
            newConv.weight.copy_(meanWeight.repeat(1, 4, 1, 1))

        self.resnet.conv1 = newConv

        self.representation = None
        self.resnet.layer4.register_forward_hook(self._save_activations)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

        if not config.trainEncoder:
            self.resnet.requires_grad_(False)
            self.resnet.conv1.requires_grad_(True)
            self.resnet.layer4.requires_grad_(True)
            self.resnet.fc.requires_grad_(True)

    def _save_activations(self, module, inputs, outputs):
        if torch.is_grad_enabled():
            self.representation = outputs
            self.representation.retain_grad()

    def preprocess(self, images):
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std  = images.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (images - mean) / std

    def forward(self, images):
        x = self.preprocess(images)
        return self.resnet(x)