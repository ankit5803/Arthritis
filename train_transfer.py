import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.densenet import KneeOADenseNet
from src.dataloader_transfer import (
    train_loader,
    val_loader
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


model = KneeOADenseNet().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=3e-5
)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=2,
    factor=0.5
)


# MIXED PRECISION
scaler = torch.cuda.amp.GradScaler()


best_val_acc = 0


for epoch in range(20):

    model.train()

    running_loss = 0
    correct = 0
    total = 0


    progress = tqdm(train_loader)


    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()


        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_acc = 100 * correct / total


    # VALIDATION
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


    scheduler.step(val_acc)


    print("\nEpoch:", epoch + 1)
    print("Loss:", running_loss)
    print("Train Accuracy:", train_acc)
    print("Validation Accuracy:", val_acc)


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved.")