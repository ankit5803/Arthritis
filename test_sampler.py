from collections import Counter

from src.dataloader import train_loader


print("\n" + "=" * 50)
print("TESTING SAMPLER")
print("=" * 50)


# get first batch
images, labels = next(iter(train_loader))


print("\nBatch size:", len(labels))


print("\nLabels in first batch:\n")

print(labels)


# count labels
label_count = Counter(labels.tolist())


print("\nClass distribution in this batch:\n")

print(label_count)