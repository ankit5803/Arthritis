import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter

from src.dataset import (
    KneeOADataset,
    train_transform,
    val_transform
)


# =========================================
# CREATE DATASETS
# =========================================
train_dataset = KneeOADataset(
    root_dir="./dataset/train",
    transform=train_transform
)

val_dataset = KneeOADataset(
    root_dir="./dataset/val",
    transform=val_transform
)


# =========================================
# CALCULATE CLASS COUNTS
# =========================================
class_counts = Counter(train_dataset.labels)



# =========================================
# CALCULATE CLASS WEIGHTS
# =========================================
class_weights = {}

for label, count in class_counts.items():

    class_weights[label] = 1.0 / count




# =========================================
# ASSIGN WEIGHT TO EACH IMAGE
# =========================================
sample_weights = []

for label in train_dataset.labels:

    sample_weights.append(
        class_weights[label]
    )


# =========================================
# CREATE SAMPLER
# =========================================
sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)


# =========================================
# DATALOADER
# =========================================
train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    sampler=sampler
)


val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False
)


if __name__ == "__main__":
    print("\nClass counts:\n")
    print(class_counts)
    print("\nClass weights:\n")
    print(class_weights)
    print("\nDataLoader created successfully.")
    

