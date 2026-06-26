import os
import cv2
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


# -----------------------------
# CLAHE FUNCTION
# -----------------------------
def apply_clahe(image):

    img_np = np.array(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(img_np)

    return Image.fromarray(enhanced)


# -----------------------------
# TRAIN TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([

    transforms.Lambda(
        lambda img: apply_clahe(img)
    ),

    transforms.RandomRotation(8),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.45],
        std=[0.22]
    )
])


# -----------------------------
# VALIDATION / TEST
# -----------------------------
val_transform = transforms.Compose([

    transforms.Lambda(
        lambda img: apply_clahe(img)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.45],
        std=[0.22]
    )
])


class KneeOADataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for label in sorted(os.listdir(root_dir)):

            class_path = os.path.join(
                root_dir,
                label
            )

            if os.path.isdir(class_path):

                for img_name in os.listdir(class_path):

                    if img_name.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    ):

                        img_path = os.path.join(
                            class_path,
                            img_name
                        )

                        self.image_paths.append(
                            img_path
                        )

                        self.labels.append(
                            int(label)
                        )


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        label = self.labels[idx]

        image = Image.open(
            img_path
        ).convert("L")


        if self.transform:

            image = self.transform(
                image
            )

        return image, label