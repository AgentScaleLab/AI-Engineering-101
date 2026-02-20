"""
Optimized ResNet18 model implementation with efficiency improvements
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as tv_resnet18


class OptimizedResNet18(nn.Module):
    """
    Wrapper for torchvision ResNet18 with potential optimizations
    """
    def __init__(self, num_classes=1000, pretrained=False):
        super(OptimizedResNet18, self).__init__()
        self.model = tv_resnet18(pretrained=pretrained)

        # Replace the final classifier layer for different number of classes if needed
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_resnet18(num_classes=1000, pretrained=False):
    """
    Factory function to get ResNet18 model
    """
    return OptimizedResNet18(num_classes=num_classes, pretrained=pretrained)