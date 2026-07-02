import torch.nn as nn
from torchvision import models


class KneeOAModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                in_features,
                5
            )
        )

    def forward(self, x):

        return self.model(x)