import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import math


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
    z0 = torch.zeros_like(logits)

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


class GradCAM2D(nn.Module):
    def __init__(self, model: nn.Module, upsample: tuple = (240, 240)):
        super().__init__()
        self.model = model
        self.upsample = upsample

    def computeCam(self):
        tokens = self.model.representation
        assert tokens.grad is not None, "No grad on lastTokens, retain_grad() set?"

        acts = tokens
        grad = tokens.grad

        if len(tokens.shape) == 3:
            def reshape(tensor):
                tensor = tensor[:, 1:]
                B, N, C = tensor.shape
                tensor = tensor.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
                tensor = nn.functional.interpolate(tensor, size=(240, 240), mode="bilinear", align_corners=False)
                return tensor

            acts = reshape(acts)
            grad = reshape(grad)

        weights = grad.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=self.upsample, mode="bilinear", align_corners=False)

        cam = cam.squeeze(1)

        B = cam.shape[0]
        camFlat = cam.view(B, -1)
        camMin = camFlat.min(dim=1).values.view(B, 1, 1)
        camMax = camFlat.max(dim=1).values.view(B, 1, 1)

        cam = (cam - camMin) / (camMax - camMin + 1e-8)
        return cam


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
        tokens = self.model.representation
        assert tokens.grad is not None, "No grad on lastTokens, retain_grad() set?"

        print("===== HERE =====")
        print(tokens.grad.amax(), tokens.grad.amin(), tokens.grad.shape)
        print("cls grad:", tokens.grad[:, 0, :].abs().max())
        print("patch grad:", tokens.grad[:, 1:, :].abs().max())

        features  = self._tokensToVolume(tokens.detach())
        gradients = self._tokensToVolume(tokens.grad.abs())

        print(features.amin(), features.amax())
        print(gradients.amin(), gradients.amax())

        weights = gradients.mean(dim=(-3, -2, -1), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=self.upsample, mode="trilinear", align_corners=False)

        B = cam.shape[0]
        camFlat = cam.view(B, -1)
        camMin = camFlat.min(dim=1).values.view(B, 1, 1, 1, 1)
        camMax = camFlat.max(dim=1).values.view(B, 1, 1, 1, 1)

        print(camMin.squeeze(), camMax.squeeze())
        print("===== THERE =====")

        print()

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


def calculateLoss(logits, labels, masks, model, gradcam, config, alpha=0.5):
    bceLoss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.squeeze())

    if masks is None:
        return bceLoss, bceLoss, None, ExplanationMetrics(*[0 for i in range(7)])
    
    active = labels > 0.5
    activeTargets = active if config.positiveOnly else torch.ones_like(labels, dtype=torch.bool)

    if model.representation.grad is not None:
        model.representation.grad.zero_()

    score = computeScore(ExplanationScore.LOGIT_SQR, logits, activeTargets)
    score.backward(retain_graph=True)

    cam = gradcam.computeCam()
    mask = masks >= 1
    
    positiveSamples = active.any(dim=1)
    
    expLoss = explanationLoss(cam[positiveSamples], mask[positiveSamples], config.topK)

    metrics = computeExplanationMetrics(cam, mask, config)

    totalLoss = bceLoss + alpha * expLoss
    return totalLoss, bceLoss, expLoss, metrics


def generateSaliencyMaps(model, gradcam, loader, config, device):
    os.makedirs(config.saliencyDirectory, exist_ok=True)
    model.eval()

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) if type(v) != list else v for k, v in batch.items()}
        model.zero_grad()

        logits = model(batch["images"])
        
        active = batch["targets"] > 0.5
        score = computeScore(
            ExplanationScore.LOGIT_SQR,
            logits,
            active
        )

        score.backward()

        cam = gradcam.computeCam()

        # print(cam.amin(), cam.amax())

        cam = cam.detach().squeeze(1).cpu().numpy()

        names = batch["names"]

        if torch.is_tensor(names):
            names = names.tolist()

        for n, name in enumerate(names):
            np.save(os.path.join(config.saliencyDirectory, f"{name}_saliency.npy"), cam[n])

        print(f"\rSaved {i+1}/{len(loader)}", end="")


@dataclass
class ExplanationMetrics:
    topSaliencyPrecision: float
    allSaliencyPrecision: float
    topSaliencyRecall: float
    allSaliencyRecall: float
    topSaliencyF1: float
    allSaliencyF1: float
    annotationCoverage: float


@torch.no_grad()
def computeExplanationMetrics(cam, mask, config):
    B = cam.shape[0]
    camFlat = cam.view(B, -1)
    maskFlat = mask.view(B, -1).float()

    threshold = torch.quantile(camFlat, 1.0 - config.topK, dim=1, keepdim=True)
    hPlus = (camFlat >= threshold).float()
    
    truePositives = (hPlus * maskFlat).sum(dim=1)
    falsePositives = (hPlus * (1.0 - maskFlat)).sum(dim=1)
    falseNegatives = ((1.0 - hPlus) * maskFlat).sum(dim=1)

    topPrecision = truePositives / (truePositives + falsePositives + 1e-8)
    topRecall = truePositives / (truePositives + falseNegatives + 1e-8)    
    topF1 = (2 * topPrecision * topRecall) / (topPrecision + topRecall + 1e-8)

    truePositives = (camFlat * maskFlat).sum(dim=1)
    falsePositives = (camFlat * (1.0 - maskFlat)).sum(dim=1)
    falseNegatives = ((1.0 - camFlat) * maskFlat).sum(dim=1)

    allPrecision = truePositives / (truePositives + falsePositives + 1e-8)
    allRecall = truePositives / (truePositives + falseNegatives + 1e-8)    
    allF1 = (2 * allPrecision * allRecall) / (allPrecision + allRecall + 1e-8)

    maskSize = maskFlat.sum(dim=1).clamp(min=1)
    overlap = (hPlus * maskFlat).sum(dim=1)
    covered = (overlap / maskSize >= config.tau).float()
    coverage = covered.mean().item()

    return ExplanationMetrics(
        topPrecision.mean().item(),
        allPrecision.mean().item(),
        topRecall.mean().item(),
        allRecall.mean().item(),
        topF1.mean().item(),
        allF1.mean().item(),
        coverage
    )