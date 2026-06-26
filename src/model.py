import torch
import torch.nn as nn


class KneeOACNN(nn.Module):

    def __init__(self):

        super().__init__()

        # BLOCK 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # BLOCK 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # BLOCK 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.dropout = nn.Dropout(0.3)

        # after pooling:
        # 224 → 112 → 56 → 28

        self.fc1 = nn.Linear(
            128 * 28 * 28,
            256
        )

        self.fc2 = nn.Linear(
            256,
            5
        )


    def forward(self, x):

        # BLOCK 1
        x = self.pool(
            self.relu(
                self.conv1(x)
            )
        )

        # BLOCK 2
        x = self.pool(
            self.relu(
                self.conv2(x)
            )
        )

        # BLOCK 3
        x = self.pool(
            self.relu(
                self.conv3(x)
            )
        )

        # flatten
        x = x.view(
            x.size(0),
            -1
        )

        x = self.dropout(
            self.relu(
                self.fc1(x)
            )
        )

        x = self.fc2(x)

        return x