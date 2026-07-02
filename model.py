import timm
import torch.nn as nn


class KneeModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = timm.create_model(

            "efficientnet_b3",

            pretrained=True,

            num_classes=5
        )

    def forward(self,x):

        return self.model(x)