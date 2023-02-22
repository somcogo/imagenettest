import torch.nn as  nn
import math
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)        

    def forward(self, x):
        out = self.model(x)
        return out