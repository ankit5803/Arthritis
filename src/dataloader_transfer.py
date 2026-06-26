from collections import Counter

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from src.dataset import (
    KneeOADataset,
    train_transform,
    val_transform
)

TRAIN_PATH = "./dataset/train"
VAL_PATH = "./dataset/val"
TEST_PATH = "./dataset/test"

train_dataset = KneeOADataset(
    TRAIN_PATH,
    transform=train_transform
)

val_dataset = KneeOADataset(
    VAL_PATH,
    transform=val_transform
)

test_dataset = KneeOADataset(
    TEST_PATH,
    transform=val_transform
)

class_counts = Counter(train_dataset.labels)

print("\nClass counts:")
print(class_counts)

weights = {
    cls: 1.0 / count
    for cls, count in class_counts.items()
}

sample_weights = [
    weights[label]
    for label in train_dataset.labels
]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nTransfer DataLoader created.")