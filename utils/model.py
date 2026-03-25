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

        self.encoder = AutoModel.from_pretrained(config.encoder, trust_remote_code=True)
        self.numPatches = (config.imageSize * config.imageSize * 32) // (16 * 16 * 4)

        if config.outputs == "segmentation":
            self.decoder = nn.LazyConv3d(config.labels, 3, 1, padding="same")
        elif config.outputs == "regressional":
            self.decoder = nn.LazyLinear(1)

    def preprocess(self, volume):
        # TODO: Determine whether this projection is reasonable, consider sum-to-one to keep "imageness"
        x = self.modalityProjection(volume)
        x = nn.functional.interpolate(x, size=(32, 256, 256), mode="trilinear", align_corners=False)
        # TODO: Normalize x
        return x

    def forward(self, volume):
        x = self.preprocess(volume)

        # Don't use the encode_image function to prevent normalization
        x, _ = self.encoder.vision_encoder(x)
        x = self.encoder.mm_vision_proj(x)

        if self.config.outputs == "segmentation":
            # TODO: Reshaping of tokens
            pass
        elif self.config.outputs == "regressional":
            x = self.decoder(x[:, 0])

        return x

