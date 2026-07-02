import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import (
    KneeDataset,
    train_transform,
    val_transform
)

from model import KneeModel


device = torch.device("cuda")


train_dataset = KneeDataset(
    "./dataset/train",
    train_transform
)

val_dataset = KneeDataset(
    "./dataset/val",
    val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)


model = KneeModel().to(device)


criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


optimizer = optim.AdamW(

    model.parameters(),

    lr=1e-4
)


best_acc = 0


for epoch in range(40):

    model.train()

    correct = 0
    total = 0
    loss_sum = 0


    progress = tqdm(train_loader)


    for images,labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        _,pred = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            pred==labels
        ).sum().item()


    train_acc = 100*correct/total


    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,pred = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                pred==labels
            ).sum().item()


    val_acc = 100*val_correct/val_total


    print("\nEpoch",epoch+1)
    print("Train:",train_acc)
    print("Val:",val_acc)


    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("saved")