import random

import numpy as np
import torch
from torchvision.transforms import (
     functional,
     RandomResizedCrop,
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing)

def aug_image(batch: torch.Tensor, mode, multi_training):
    if multi_training:
        n, b, c, h, w = batch.shape
        batch = batch.reshape(-1, c, h, w)
    if mode == 'standard':
        batch = aug_flip_rotate_scale(batch)
    elif mode == 'random_resized_crop':
        batch = aug_flip_resized_crop(batch)
    elif mode == 'resnet':
        batch = aug_crop_rotate_flip_erase(batch)
    if multi_training:
        batch = batch.reshape(n, b, c, h, w)
    return batch

def aug_flip_rotate_scale(batch):
    flip = random.choice([True, False])
    angle = random.choice([0, 90, 180, 270])
    scale = random.uniform(0.9, 1.1)
    if flip:
        batch = functional.hflip(batch)
    batch = functional.rotate(batch, angle)
    batch = scale * batch
    return batch

def aug_flip_resized_crop(batch):
    flip = random.choice([True, False])
    if flip:
        batch = functional.hflip(batch)
    resized_crop = RandomResizedCrop(64)
    batch = resized_crop(batch)

def aug_crop_rotate_flip_erase(batch):
    trans = Compose([
        Pad(4),
        RandomCrop(64),
        RandomHorizontalFlip(p=0.25),
        RandomApply(torch.nn.ModuleList([
            RandomRotation(degrees=15)
        ]), p=0.25),
        RandomErasing(p=0.5, scale=(0.015625, 0.25), ratio=(0.25, 4))
    ])
    batch = trans(batch)
    return batch