from torchvision import datasets
from torchvision.transforms import functional
from torch.utils.data import DataLoader, Dataset
import h5py
import os
import numpy as np
import torch

data_path = 'data'

class DataSet(Dataset):
    def __init__(self, data_path, mode, device,site=None):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_path, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
        self.image_ds = h5_file['data']
        if mode != 'tst':
            self.labels = h5_file['labels']
        if site is not None:
            self.image_ds = self.image_ds[site * 20000: (site + 1) * 20000]
            self.labels = self.labels[site * 20000: (site + 1) * 20000]

        self.angles = [0, 90, 180, 270]
        self.mode = mode
        self.device = device
    def __len__(self):
        return self.image_ds.shape[0]

    def __getitem__(self, index):
        image_np = np.array(self.image_ds[index])
        image = torch.from_numpy(image_np).permute(2, 0, 1)

        if self.labels is not None:
            label_np = np.array(self.labels[index])
            label = torch.from_numpy(label_np)
            return image, label
        else:
            return image

def get_trn_loader(batch_size, device, site=None):
    trn_dataset = DataSet(data_path=data_path, mode='trn', device=device,site=site)
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader

def get_val_loader(batch_size, device):
    val_dataset = DataSet(data_path=data_path, mode='val', device=device)
    train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return train_loader

def get_tst_loader(batch_size, device):
    tst_dataset = DataSet(data_path=data_path, mode='tst', device=device)
    train_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader