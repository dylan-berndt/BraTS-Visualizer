from utils import *

config = Config().load(os.path.join("configs", "config.json"))
dataset = BraTSData(config)

model = BraTSM3D(config).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

objective = nn.BCEWithLogitsLoss()

for epoch in range(10):
    for batch in DataLoader(dataset, batch_size=4, shuffle=True):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(batch)

        loss = objective(outputs, batch["targets"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\rEpoch {epoch} | Loss: {loss.item()}", end="")

