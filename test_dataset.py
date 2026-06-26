from src.dataset import KneeOADataset, train_transform


dataset = KneeOADataset(
    root_dir="./dataset/train",
    transform=train_transform
)


print("\n" + "=" * 50)
print("TESTING DATASET")
print("=" * 50)


print("Dataset size:", len(dataset))


image, label = dataset[0]


print("\nFirst Sample Information:\n")

print("Image shape:", image.shape)

print("Label:", label)

print("Tensor type:", image.dtype)

print("Min pixel value:", image.min())

print("Max pixel value:", image.max())