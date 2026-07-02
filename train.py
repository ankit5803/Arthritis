import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.model import KneeOAModel
from src.dataloader import (
    train_loader,
    val_loader
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


model = KneeOAModel().to(device)


criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)


scaler = torch.amp.GradScaler("cuda")


best_val_acc = 0
patience = 6
counter = 0


for epoch in range(50):

    model.train()

    correct = 0
    total = 0
    loss_sum = 0

    progress = tqdm(train_loader)

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        loss_sum += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_acc = 100 * correct / total


    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print("\nEpoch", epoch + 1)
    print("Train:", train_acc)
    print("Val:", val_acc)


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        counter = 0

        print("saved")

    else:

        counter += 1


    if counter >= patience:

        print("Early stopping")

        break