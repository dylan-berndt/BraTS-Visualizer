from utils import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import json
import os

DEVICE = "cuda:1"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ENCODER_FROZEN = [True, False]
MODELS = ["ResNet50", "ViT"]

class AUC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        yPred = nn.functional.sigmoid(yPred)
        auroc = roc_auc_score(
            yTrue.cpu().detach().numpy().ravel(),
            yPred.cpu().detach().numpy().ravel()
        )
        return auroc


def getRunName(modelName, alpha, frozen):
    encoderStr = "Frozen" if frozen else "Unfrozen"
    return f"{modelName} {alpha} {encoderStr}"


def buildModel(modelName, config, device):
    if modelName == "ResNet50":
        model = BraTSNet50(config).to(device)
        gradcam = GradCAM2D(model)
    else:
        model = BraTSViT(config).to(device)
        gradcam = GradCAM2D(model)
    return model, gradcam


def runEpoch(model, gradcam, loader, optimizer, config, alpha, device, auc, train=True):
    if train:
        model.train()
    else:
        model.eval()

    totalLossAcc = 0
    allLogits = []
    allTargets = []
    allPreds = []

    topPrecisionAcc = 0
    allPrecisionAcc = 0
    topRecallAcc = 0
    allRecallAcc = 0
    topF1Acc = 0
    allF1Acc = 0
    coverageAcc = 0
    nBatches = 0

    context = torch.enable_grad if train else torch.no_grad

    with context():
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) if type(v) != list else v for k, v in batch.items()}

            if train:
                optimizer.zero_grad()

            logits = model(batch["images"])

            if train:
                totalLoss, bceLoss, expLoss, expMetrics = calculateLoss(
                    logits, batch["targets"], batch["masks"], model, gradcam, config, alpha=alpha
                )
                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()
                totalLossAcc += totalLoss.item()

                topPrecisionAcc += expMetrics.topSaliencyPrecision
                allPrecisionAcc += expMetrics.allSaliencyPrecision
                topRecallAcc += expMetrics.topSaliencyRecall
                allRecallAcc += expMetrics.allSaliencyRecall
                topF1Acc += expMetrics.topSaliencyF1
                allF1Acc += expMetrics.allSaliencyF1
                coverageAcc += expMetrics.annotationCoverage
            else:
                bceLoss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(), batch["targets"].squeeze()
                )
                totalLossAcc += bceLoss.item()

            allLogits.append(logits.detach())
            allTargets.append(batch["targets"].detach())
            probs = torch.sigmoid(logits.detach())
            allPreds.append((probs > 0.5).long())
            nBatches += 1

            print(f"\r  {'Train' if train else 'Test'} batch {b + 1}/{len(loader)}", end="")

    allLogits = torch.cat(allLogits)
    allTargets = torch.cat(allTargets)
    allPreds = torch.cat(allPreds).cpu().numpy()
    targetsNp = allTargets.cpu().numpy().ravel()

    epochAUC = auc(allLogits, allTargets)
    epochPrecision = precision_score(targetsNp, allPreds, zero_division=0)
    epochRecall = recall_score(targetsNp, allPreds, zero_division=0)
    epochF1 = f1_score(targetsNp, allPreds, zero_division=0)
    epochLoss = totalLossAcc / nBatches

    metrics = {
        "loss": epochLoss,
        "auc": epochAUC,
        "precision": epochPrecision,
        "recall": epochRecall,
        "f1": epochF1,
    }

    if train:
        metrics["topSaliencyPrecision"] = topPrecisionAcc / nBatches
        metrics["allSaliencyPrecision"] = allPrecisionAcc / nBatches
        metrics["topSaliencyRecall"] = topRecallAcc / nBatches
        metrics["allSaliencyRecall"] = allRecallAcc / nBatches
        metrics["topSaliencyF1"] = topF1Acc / nBatches
        metrics["allSaliencyF1"] = allF1Acc / nBatches
        metrics["annotationCoverage"] = coverageAcc / nBatches

    return metrics


def trainRun(modelName, alpha, frozen, config, trainSet, testSet, device):
    runName = getRunName(modelName, alpha, frozen)
    runDir = os.path.join("checkpoints", runName)
    os.makedirs(runDir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Run: {runName}")
    print(f"{'='*60}")

    config.trainEncoder = not frozen
    model, gradcam = buildModel(modelName, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate)
    auc = AUC()

    trainLoader = DataLoader(trainSet, batch_size=config.batchSize, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=config.batchSize, shuffle=False)

    allMetrics = []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        trainMetrics = runEpoch(
            model, gradcam, trainLoader, optimizer, config, alpha, device, auc, train=True
        )
        testMetrics = runEpoch(
            model, gradcam, testLoader, optimizer, config, alpha, device, auc, train=False
        )

        epochLog = {
            "epoch": epoch + 1,
            "train": trainMetrics,
            "test": testMetrics,
        }
        allMetrics.append(epochLog)

        print(
            f"\r  Epoch {epoch + 1} | "
            f"Train Loss: {trainMetrics['loss']:.4f} | Train AUC: {trainMetrics['auc']:.4f} | "
            f"Test Loss: {testMetrics['loss']:.4f} | Test AUC: {testMetrics['auc']:.4f} | "
            f"Test Precision: {testMetrics['precision']:.4f} | Test Recall: {testMetrics['recall']:.4f} | "
            f"Test F1: {testMetrics['f1']:.4f}"
        )

    # Save model
    modelPath = os.path.join(runDir, "model.pt")
    torch.save(model.state_dict(), modelPath)
    print(f"\nSaved model to {modelPath}")

    # Save metrics
    metricsPath = os.path.join(runDir, "metrics.json")
    with open(metricsPath, "w") as f:
        json.dump(allMetrics, f, indent=2)
    print(f"Saved metrics to {metricsPath}")

    # Generate saliency maps on test set
    saliencyDir = os.path.join(runDir, "saliency")
    config.saliencyDirectory = saliencyDir
    print(f"\nGenerating saliency maps to {saliencyDir}...")
    generateSaliencyMaps(model, gradcam, testLoader, config=config, device=device)
    print(f"\nSaliency maps saved to {saliencyDir}")



def main():
    config = Config().load(os.path.join("configs", "resnetConfig.json"))
    dataset = BraTSData(config)

    trainSet, testSet = torch.utils.data.random_split(dataset, [0.8, 0.2])

    for frozen in ENCODER_FROZEN:
        for modelName in MODELS:
            for alpha in ALPHAS:
                trainRun(modelName, alpha, frozen, config, trainSet, testSet, DEVICE)


if __name__ == "__main__":
    main()