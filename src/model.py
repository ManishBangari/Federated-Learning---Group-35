"""
model.py — Neural network architectures.

Available models:
  - LeNet5      : for MNIST, FashionMNIST (1-channel, 28x28)
  - SimpleCNN   : lightweight CNN for MNIST/FMNIST
  - ResNet18    : for CIFAR-10, CIFAR-100 (3-channel, 32x32)

Weight initialization: PyTorch default (as per FedAvg spec).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────
#  LeNet-5  (MNIST / FashionMNIST)
# ──────────────────────────────────────────────

class LeNet5(nn.Module):
    """
    Classic LeNet-5 adapted for 28x28 grayscale inputs.
    Used for MNIST and FashionMNIST.
    """
    def __init__(self, num_classes: int = 10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28→28
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)        # 28→14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)              # 14→10
        # after pool: 10→5
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B,6,14,14)
        x = self.pool(F.relu(self.conv2(x)))   # (B,16,5,5)
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ──────────────────────────────────────────────
#  SimpleCNN  (MNIST / FashionMNIST — lighter)
# ──────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Lightweight 2-block CNN for 28x28 grayscale inputs.
    Faster to train, useful for large client counts.
    """
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                     # 28→14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                     # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
#  ResNet-18  (CIFAR-10 / CIFAR-100)
# ──────────────────────────────────────────────

class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR (32x32, 3-channel).
    Replaces the first conv layer to suit small images.
    Uses PyTorch default weight initialization.
    """
    def __init__(self, num_classes: int = 10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)  # no pretrained weights

        # Replace conv1: 7x7/stride-2 → 3x3/stride-1 (better for 32x32)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove maxpool (not helpful for small images)
        self.model.maxpool = nn.Identity()

        # Replace final FC layer
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ──────────────────────────────────────────────
#  Model Factory
# ──────────────────────────────────────────────

def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    """
    Return the requested model with default PyTorch weight initialization.

    Args:
        architecture : one of 'lenet', 'simplecnn', 'resnet18'
        num_classes  : output dimension

    Returns:
        nn.Module
    """
    architecture = architecture.lower()

    if architecture == "lenet":
        model = LeNet5(num_classes=num_classes)

    elif architecture == "simplecnn":
        model = SimpleCNN(num_classes=num_classes)

    elif architecture == "resnet18":
        model = ResNet18(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Choose from: lenet, simplecnn, resnet18"
        )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {architecture.upper()} loaded — "
          f"Params: {total_params:,} | Classes: {num_classes}")
    return model


# ──────────────────────────────────────────────
#  Parameter extraction (used by main.py + server.py)
# ──────────────────────────────────────────────

def get_parameters(model: nn.Module):
    """Extract model weights as list of numpy arrays."""
    import numpy as np
    return [val.cpu().numpy() for _, val in model.state_dict().items()]