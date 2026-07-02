import os
import random
from PIL import Image
from torchvision import transforms

# ==========================
# SETTINGS
# ==========================
CLASS_FOLDER = "./dataset/train/1"     # grade 1 folder
TARGET_COUNT = 2000                   # total images wanted

# ==========================
# AUGMENTATIONS
# ==========================
augment = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    )

])

# ==========================
# GET EXISTING IMAGES
# ==========================
images = [
    f for f in os.listdir(CLASS_FOLDER)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

current_count = len(images)

print("Current images:", current_count)

needed = TARGET_COUNT - current_count

if needed <= 0:
    print("Already enough images.")
    exit()

print("Need to create:", needed)

# ==========================
# GENERATE NEW IMAGES
# ==========================
created = 0

while created < needed:

    img_name = random.choice(images)

    img_path = os.path.join(
        CLASS_FOLDER,
        img_name
    )

    image = Image.open(
        img_path
    ).convert("RGB")

    aug_image = augment(image)

    new_name = f"aug_{created}_{img_name}"

    save_path = os.path.join(
        CLASS_FOLDER,
        new_name
    )

    aug_image.save(save_path)

    created += 1

    if created % 100 == 0:
        print("Created:", created)

print("Done.")
print("Final count:", TARGET_COUNT)