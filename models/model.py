import torch.nn as  nn
import torch
from torchvision.models import resnet18, ResNet18_Weights, swin_t, swin_s
from timm import create_model
from torchvision.transforms import Resize

from .UNetBlock  import UNetBlock

class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)        

    def forward(self, x):
        out = self.model(x)
        return out

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1)
        self.block1 = UNetBlock(in_channels=4, out_channels=64, down=True, attention=True)
        self.block2 = UNetBlock(in_channels=64, out_channels=128, down=True, attention=True)
        self.block3 = UNetBlock(in_channels=128, out_channels=256, down=True, attention=True)
        self.block4 = UNetBlock(in_channels=256, out_channels=512, down=True, attention=True)
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.lin = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        out = self.lin(x)
        return out

class TinySwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.tiny_swin = swin_t(num_classes=num_classes)

    def forward(self, x):
        out = self.tiny_swin(x)
        return out
    
class SmallSwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.small_swin = swin_s(num_classes=num_classes)

    def forward(self, x):
        out = self.small_swin(x)
        return out

class SmallSwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.small_swin = swin_s(num_classes=num_classes)

    def forward(self, x):
        out = self.small_swin(x)
        return out

class LargeSwin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.large_swin = create_model('swin_large_patch4_window12_384',
                                     pretrained=False, drop_path_rate=0.1,
                                     num_classes=num_classes)
        self.resize = Resize(size=(384, 384), antialias=True)

    def forward(self, x):
        x = self.resize(x)
        out = self.large_swin(x)
        return out