from utils import *
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:1"

config = Config().load(os.path.join("configs", "config.json"))
dataset = BraTSData(config)

model = BraTSM3D(config).to(DEVICE)
gradcam = GradCAM3D(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


class AUC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        yPred = nn.functional.sigmoid(yPred)
        auroc = roc_auc_score(yTrue.cpu().detach().numpy().ravel(), yPred.cpu().detach().numpy().ravel())

        return auroc
    

auc = AUC()

trainSet, testSet = torch.utils.data.random_split(dataset, [0.8, 0.2])

for epoch in range(1):
    model.train()
    train = DataLoader(trainSet, batch_size=4, shuffle=True)
    for b, batch in enumerate(train):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()

        logits = model(batch["images"])

        totalLoss, bceLoss, expLoss = calculateLoss(logits, batch["targets"], batch["masks"], model, gradcam, config)
        totalLoss.backward()
        optimizer.step()

        score = auc(logits, batch["targets"])

        print(f"\rEpoch {epoch + 1} | {b + 1}/{len(train)} | Train Loss: {totalLoss.item():.2f} | Train AUC: {score:.2f}", end="")

    test = DataLoader(testSet, batch_size=4, shuffle=True)
    testAverageLoss = 0
    testAverageAUC = 0
    testTotal = 0
    with torch.no_grad():
        model.eval()
        for b, batch in enumerate(test):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["images"])

            bceLoss = F.binary_cross_entropy_with_logits(logits, batch["targets"])

            score = auc(logits, batch["targets"])

            total = batch["images"].shape[0]
            testAverageLoss = (testAverageLoss * testTotal + bceLoss.item() * total) / (testTotal + total)
            testAverageAUC = (testAverageAUC * testTotal + score * total) / (testTotal + total)
            testTotal += total

            print(f"\rEpoch {epoch + 1} | {b + 1}/{len(test)} | Test Loss: {bceLoss.item():.2f} | Test AUC: {score:.2f}", end="")

    print(f"\rEpoch {epoch + 1} | Test Loss: {testAverageLoss:.2f} | Test AUC: {testAverageAUC:.2f} {' ' * 50}")

    print("\nExporting saliency maps...")

    generateSaliencyMaps(
        model,
        test,
        config=config,
        device=DEVICE
    )
