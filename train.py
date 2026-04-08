from utils import *
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:1"

config = Config().load(os.path.join("configs", "config.json"))
dataset = BraTSData(config)

model = BraTSM3D(config).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

objective = nn.BCEWithLogitsLoss()


class AUC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        yPred = nn.functional.sigmoid(yPred)
        auroc = roc_auc_score(yTrue.cpu().detach().numpy().ravel(), yPred.cpu().detach().numpy().ravel())

        return auroc
    

auc = AUC()

trainSet, testSet = torch.utils.data.random_split(dataset, [0.8, 0.2])

for epoch in range(10):
    train = DataLoader(trainSet, batch_size=4, shuffle=True)
    for b, batch in enumerate(train):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(batch)

        loss = objective(outputs, batch["targets"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = auc(outputs, batch["targets"])

        print(f"\rEpoch {epoch} | {b}/{len(train)} | Train Loss: {loss.item()} | Train AUC: {score}", end="")

    test = DataLoader(testSet, batch_size=8, shuffle=True)
    for b, batch in enumerate(test):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(batch)

        loss = objective(outputs, batch["targets"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = auc(outputs, batch["targets"])

        print(f"\rEpoch {epoch} | {b}/{len(train)} | Test Loss: {loss.item()} | Test AUC: {score}", end="")
