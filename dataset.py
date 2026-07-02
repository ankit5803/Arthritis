import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


train_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.RandomHorizontalFlip(0.5),

    transforms.RandomVerticalFlip(0.3),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05,0.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


class KneeDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.images = []
        self.labels = []
        self.transform = transform

        for label in sorted(os.listdir(root_dir)):

            folder = os.path.join(root_dir,label)

            for file in os.listdir(folder):

                if file.endswith((".png",".jpg",".jpeg")):

                    self.images.append(
                        os.path.join(folder,file)
                    )

                    self.labels.append(
                        int(label)
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = Image.open(
            self.images[idx]
        ).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img,label