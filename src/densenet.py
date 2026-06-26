import torch
import torch.nn as nn
from torchvision import models


class KneeOADenseNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )


        self.model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )


        for param in self.model.parameters():
            param.requires_grad = False


        for param in self.model.features.denseblock4.parameters():
            param.requires_grad = True


        for param in self.model.features.norm5.parameters():
            param.requires_grad = True


        # IMPROVED CLASSIFIER
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 5)
        )


        for param in self.model.classifier.parameters():
            param.requires_grad = True


    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = KneeOADenseNet()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", trainable)