import torch
import torch.nn.functional as F
from torch import nn

from models.safe_region.safe_region import SafeRegion2d

__all__ = ['simplemodel']

class SimpleModel(nn.Module):
    def __init__(self, num_channels, batch_norm, safe_region, num_classes):
        super().__init__()

        layers = []

        for channels in num_channels:
            layers.extend(self._make_layer(channels, batch_norm, safe_region))

        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(num_channels[-1] * 2 * 2, num_classes)


    def _make_layer(self, num_channels, batch_norm=True, safe_region=True):
        layer = []

        layer.append(nn.Conv2d(3, num_channels, 3, 1, 1))
        if batch_norm:
            layer.append(nn.BatchNorm2d(num_channels))
        if safe_region:
            layer.append(SafeRegion2d(num_channels))
        layer.append(nn.ReLU())
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return layer

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

def simplemodel(num_channels=[64,128,256,512], batch_norm=True, safe_region=True, num_classes=10):
    return SimpleModel(num_channels, batch_norm, safe_region, num_classes)