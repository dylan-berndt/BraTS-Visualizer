import torch
import torch.nn as nn
from transformers import AutoModel

from .config import *


class BraTSM3D(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        # Effectively nn.Linear but idc
        self.modalityProjection = nn.Conv3d(4, 1, 1)

        print("DEFAULT DEVICE:", torch.get_default_device())

        self.encoder = AutoModel.from_pretrained(config.encoder, trust_remote_code=True, device_map=None, low_cpu_mem_usage=False, _fast_init=False)
        self.encoder.requires_grad_(config.trainEncoder)
        self.numPatches = (config.imageSize * config.imageSize * 32) // (16 * 16 * 4)

        if config.outputs == "segmentation":
            self.transpose = nn.LazyConvTranspose3d(32, kernel_size=(4, 16, 16), stride=(4, 16, 16))
            self.decoder = nn.LazyConv3d(config.labels, 3, 1, padding="same")
        elif config.outputs == "regressional":
            self.decoder = nn.LazyLinear(1)
        elif config.outputs == "categorical":
            self.decoder = nn.LazyLinear(config.targets)

    def preprocess(self, volume):
        # TODO: Determine whether this projection is reasonable, consider sum-to-one to keep "imageness"
        x = volume.permute(0, 2, 1, 3, 4)
        x = self.modalityProjection(x)
        x = nn.functional.interpolate(x, size=(32, 256, 256), mode="trilinear", align_corners=False)

        p1 = torch.quantile(x.float(), 0.01)
        p99 = torch.quantile(x.float(), 0.99)
        x = x.clamp(p1, p99)

        # TODO: Check normalization
        x = x - x.amin(dim=(-3, -2, -1), keepdim=True)
        x = x / (x.amax(dim=(-3, -2, -1), keepdim=True) + 1e-5)
        return x

    def forward(self, inputs):
        x = self.preprocess(inputs)

        # Don't use the encode_image function to prevent normalization
        x, states = self.encoder.vision_encoder(x)
        x = self.encoder.mm_vision_proj(x)

        # Extract hidden state before final attention, allows for gradient flow through CLS
        if torch.is_grad_enabled():
            for i in range(len(states)):
                states[i].retain_grad()
            self.allTokens = states
            lastHidden = states[-2]
            lastHidden.retain_grad()
            self.lastTokens = lastHidden

        if self.config.outputs == "segmentation":
            B, N, C = x.shape
            # Remove cls token
            x = x[:, 1:]
            x = x.view(B, 8, 16, 16, C)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.transpose(x)
            x = self.decoder(x.unsqueeze(1))

        elif self.config.outputs == "regressional":
            x = self.decoder(x[:, 0])

        elif self.config.outputs == "categorical":
            x = self.decoder(x[:, 0])

        return x

