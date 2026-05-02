import numpy as np
import torch
from torch import nn
import torchvision

class CNNModel(nn.Module):
    def __init__(
        self,
        dropout_p,
    ):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 16, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),

            # Bloque 2
            nn.Conv2d(16, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),

            # Bloque 3
            nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            # El primer valor de esta capa lineal cambia si se utilizan distintos bloques
            nn.Linear(2048, 50),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(50, 3),
        )

        self.net = nn.Sequential(
            self.conv_blocks,
            self.mlp,
        )

    def forward(self, x):
        return self.net(x)