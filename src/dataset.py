import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


train_transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(8),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03)
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class KneeOADataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for label in sorted(os.listdir(root_dir)):

            class_path = os.path.join(root_dir, label)

            if os.path.isdir(class_path):

                for img_name in os.listdir(class_path):

                    if img_name.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    ):

                        self.image_paths.append(
                            os.path.join(class_path, img_name)
                        )

                        self.labels.append(
                            int(label)
                        )

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        label = self.labels[idx]

        if self.transform:

            image = self.transform(image)

        return image, label