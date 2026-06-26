import torch
import torch.nn as nn

from src.model import KneeOACNN
from src.dataloader import train_loader, val_loader


# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# MODEL
model = KneeOACNN().to(device)


# LOSS FUNCTION
criterion = nn.CrossEntropyLoss()


# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# EPOCHS
epochs = 20


for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    # TRAIN MODE
    model.train()

    running_loss = 0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # clear old gradients
        optimizer.zero_grad()


        # forward pass
        outputs = model(images)


        # calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # backprop
        loss.backward()


        # update weights
        optimizer.step()


        running_loss += loss.item()


        # predictions
        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * correct / total
    )


    print(
        "Loss:",
        running_loss
    )

    print(
        "Train Accuracy:",
        train_accuracy
    )