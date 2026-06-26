import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================
# DATASET ROOT
# ==========================
DATASET_ROOT = r"./dataset"

# Dataset splits
splits = ["train", "val", "test"]


# =====================================
# 1. CHECK CLASS DISTRIBUTION
# =====================================
print("\n" + "=" * 50)
print("CHECKING CLASS DISTRIBUTION")
print("=" * 50)

for split in splits:

    print(f"\n{split.upper()} SET:\n")

    split_path = os.path.join(DATASET_ROOT, split)

    for cls in sorted(os.listdir(split_path)):

        class_path = os.path.join(split_path, cls)

        if os.path.isdir(class_path):

            count = len([
                img for img in os.listdir(class_path)
                if img.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

            print(f"Grade {cls}: {count} images")


# =====================================
# 2. CHECK IMAGE SIZE DISTRIBUTION
# =====================================
print("\n" + "=" * 50)
print("CHECKING IMAGE SIZE DISTRIBUTION")
print("=" * 50)

size_dict = defaultdict(int)
corrupted = []

# Only checking train set for speed
train_path = os.path.join(DATASET_ROOT, "train")

for cls in sorted(os.listdir(train_path)):

    class_path = os.path.join(train_path, cls)

    if os.path.isdir(class_path):

        images = [
            img for img in os.listdir(class_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for img_name in images:

            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path)
                size_dict[img.size] += 1

            except Exception:
                corrupted.append(img_path)

print("\nImage Sizes Found:\n")

for size, count in size_dict.items():
    print(f"{size} --> {count} images")


# =====================================
# 3. CHECK CORRUPTED IMAGES
# =====================================
print("\n" + "=" * 50)
print("CHECKING CORRUPTED IMAGES")
print("=" * 50)

if len(corrupted) == 0:
    print("\nNo corrupted images found.")

else:
    print("\nCorrupted Images Found:\n")

    for file in corrupted:
        print(file)


# =====================================
# 4. DISPLAY SAMPLE IMAGES
# =====================================
print("\n" + "=" * 50)
print("DISPLAYING SAMPLE IMAGES")
print("=" * 50)

classes = sorted(os.listdir(train_path))

plt.figure(figsize=(15, 5))

for i, cls in enumerate(classes):

    class_path = os.path.join(train_path, cls)

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    sample = random.choice(images)

    img_path = os.path.join(class_path, sample)

    img = Image.open(img_path)

    plt.subplot(1, len(classes), i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Grade {cls}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# =====================================
# DONE
# =====================================
print("\nDataset EDA complete.\n")