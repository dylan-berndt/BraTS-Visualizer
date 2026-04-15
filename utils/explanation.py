import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np


class ExplanationScore(Enum):
    """
    All 7 score formulations from the paper.
    z0/p0 = negative class logit/prob, z1/p1 = positive class logit/prob.
    """
    LOGIT_ALG   = "ld_alg"    # z1 - z0
    LOGIT_ABS   = "ld_abs"    # |z1 - z0|
    LOGIT_SQR   = "ld_sqr"    # (z1 - z0)^2
    LOGIT_ONLY  = "l_only"    # z1 only, no contrast
    PROB_ALG    = "p_alg"     # p1 - p0
    PROB_ABS    = "p_abs"     # |p1 - p0|
    PROB_SQR    = "p_sqr"     # (p1 - p0)^2


def computeScore(formulation: ExplanationScore, logits: torch.Tensor, activeTargets: Optional[torch.Tensor] = None):
    z1 = logits
    z0 = torch.zeros_like(z1)

    if formulation == ExplanationScore.LOGIT_ONLY:
        scores = z1
    elif formulation == ExplanationScore.LOGIT_ALG:
        scores = z1 - z0
    elif formulation == ExplanationScore.LOGIT_ABS:
        scores = (z1 - z0).abs()
    elif formulation == ExplanationScore.LOGIT_SQR:
        scores = (z1 - z0) ** 2
    else:
        p1 = torch.sigmoid(z1)
        p0 = 1.0 - p1
        if formulation == ExplanationScore.PROB_ALG:
            scores = p1 - p0
        elif formulation == ExplanationScore.PROB_ABS:
            scores = (p1 - p0).abs()
        elif formulation == ExplanationScore.PROB_SQR:
            scores = (p1 - p0) ** 2
        
    if activeTargets is None:
        activeTargets = torch.ones_like(scores)
    scores = scores * activeTargets.float()

    return scores.sum()


class GradCAM3D(nn.Module):
    def __init__(self, model: nn.Module, upsample: tuple = (155, 240, 240)):
        super().__init__()
        self.model = model
        self.upsample = upsample

    def _tokensToVolume(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        x = tokens[:, 1:]
        x = x.view(B, 8, 16, 16, C)
        x = x.permute(0, 4, 1, 2, 3) 
        return x
    
    def computeCam(self):
        tokens = self.model.lastTokens
        assert tokens.grad is not None, "No grad on lastTokens — retain_grad() set?"

        features  = self._tokensToVolume(tokens.detach())
        gradients = self._tokensToVolume(tokens.grad)

        weights = gradients.mean(dim=(-3, -2, -1), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=self.upsample, mode="trilinear", align_corners=False)

        B = cam.shape[0]
        camFlat = cam.view(B, -1)
        camMin = camFlat.min(dim=1).values.view(B, 1, 1, 1, 1)
        camMax = camFlat.max(dim=1).values.view(B, 1, 1, 1, 1)
        cam = (cam - camMin) / (camMax - camMin + 1e-8)
        return cam


def explanationLoss(cam, mask, topK = 0.5):
    B = cam.shape[0]
    camFlat = cam.view(B, -1)
    maskFlat = mask.view(B, -1).float()

    threshold = torch.quantile(camFlat, 1.0 - topK, dim=1, keepdim=True)
    hPlus = (camFlat >= threshold).float()

    numerator = (hPlus * maskFlat).sum(dim=1)
    denominator = hPlus.sum(dim=1) + 1e-8

    loss = 1.0 - numerator / denominator
    return loss.mean()


def calculateLoss(logits, labels, masks, model, gradcam, config):
    bceLoss = F.binary_cross_entropy_with_logits(logits, labels)

    if config.alpha == 0.0 or masks is None:
        return bceLoss, bceLoss, None
    
    active = labels > 0.5
    activeTargets = active if config.positiveOnly else torch.ones_like(labels, dtype=torch.bool)

    if model.lastTokens.grad is not None:
        model.lastTokens.grad.zero_()

    score = computeScore(ExplanationScore.LOGIT_SQR, logits, activeTargets)
    score.backward(retain_graph=True)

    cam = gradcam.computeCam()
    mask = masks >= 1
    
    positiveSamples = active.any(dim=1)
    
    expLoss = explanationLoss(cam[positiveSamples], mask[positiveSamples], config.topK)

    totalLoss = bceLoss + config.alpha * expLoss
    return totalLoss, bceLoss, expLoss


def generateSaliencyMaps(model, loader, config, device):
    os.makedirs(config.saliencyDirectory, exist_ok=True)
    gradcam = GradCAM3D(model)
    model.eval()

    print()

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) if type(v) != list else v for k, v in batch.items()}
        logits = model(batch["images"])

        model.zero_grad()
        active = batch["targets"] > 0.5
        score = computeScore(
            ExplanationScore.LOGIT_SQR,
            logits,
            active
        )

        score.backward()

        cam = gradcam.computeCam()
        cam = cam.squeeze(1).cpu().numpy()

        names = batch["names"]

        if torch.is_tensor(names):
            names = names.tolist()

        for n, name in enumerate(names):
            np.save(os.path.join(config.saliencyDirectory, f"{name}_saliency.npy"), cam[n])

        print(f"\rSaved {i+1}/{len(loader)}", end="")