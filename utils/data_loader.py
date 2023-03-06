import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

data_path = 'data'

class ImageNetDataSet(Dataset):
    def __init__(self, data_path, mode, site=None, site_number=None):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_path, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
        self.image_ds = h5_file['data']
        if mode != 'tst':
            self.labels = h5_file['labels']
        if site is not None:
            self.image_ds = self.image_ds[site * 20000: (site + 1) * 20000]
            self.labels = self.labels[site * 20000: (site + 1) * 20000]
        if site_number is not None:
            self.site_number = site_number
            self.image_per_site = self.image_ds.shape[0] // site_number

    def __len__(self):
        if hasattr(self, 'site_number'):
            length = self.image_per_site
        else:
            length = self.image_ds.shape[0]
        return length

    def __getitem__(self, index):
        if hasattr(self, 'site_number'):
            indices = []
            for i in range(self.site_number):
                indices.append(i * self.image_per_site + index)
            images_np = np.array([self.image_ds[ndx] for ndx in indices])
            images = torch.from_numpy(images_np).permute(0, 3, 1, 2)
            labels_np = np.array([self.labels[ndx] for ndx in indices])
            labels = torch.from_numpy(labels_np)

            return images, labels
        else:
            image_np = np.array(self.image_ds[index])
            image = torch.from_numpy(image_np).permute(2, 0, 1)

            if self.labels is not None:
                label_np = np.array(self.labels[index])
                label = torch.from_numpy(label_np)
                return image, label
            else:
                return image

# class MultiSiteDataSet(Dataset):
#     def __init__(self, data_path, mode, site_number=5):
#         super().__init__()
#         h5_file = h5py.File(os.path.join(data_path, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
#         self.image_ds = h5_file['data']
#         self.labels = h5_file['labels']
#         self.site_number = site_number
#         self.image_per_site = self.image_ds.shape[0] // site_number

#     def __len__(self):
#         return self.image_per_site
    
#     def __getitem__(self, index):
#         indices = []
#         for i in range(self.site_number):
#             indices.append(i * self.image_per_site + index)
#         images_np = np.array([self.image_ds[ndx] for ndx in indices])
#         images = torch.from_numpy(images_np).permute(0, 3, 1, 2)
#         labels_np = np.array([self.labels[ndx] for ndx in indices])
#         labels = torch.from_numpy(labels_np)

#         return images, labels

def get_trn_loader(batch_size, device, site=None):
    trn_dataset = ImageNetDataSet(data_path=data_path, mode='trn', site=site)
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader

def get_multi_site_trn_loader(batch_size, site_number):
    trn_dataset = ImageNetDataSet(data_path=data_path, mode='trn', site_number=site_number)
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader

def get_val_loader(batch_size, device):
    val_dataset = ImageNetDataSet(data_path=data_path, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return val_loader

def get_multi_site_val_loader(batch_size, site_number):
    val_dataset = ImageNetDataSet(data_path=data_path, mode='val', site_number=site_number)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return val_loader

def get_tst_loader(batch_size, device):
    tst_dataset = ImageNetDataSet(data_path=data_path, mode='tst')
    test_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    return test_loader
