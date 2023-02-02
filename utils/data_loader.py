from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset
import h5py
import os
import numpy as np
import torch

data_path = 'data'

class DataSet(Dataset):
    def __init__(self, data_path, mode):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_path, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
        self.image_ds = h5_file['data']
        if h5_file['labels'] is not None:
            labels_ds = h5_file['labels']
            self.labels = labels_ds.asstr()[()]
    
    def __len__(self):
        return self.image_ds.shape[0]

    def __getitem__(self, index):
        image_np = np.array(self.image_ds[index])
        image = torch.from_numpy(image_np)
        if self.labels is not None:
            return image, self.labels
        else:
            return image
        
trn_dataset = DataSet(data_path=data_path, mode='trn')
val_dataset = DataSet(data_path=data_path, mode='val')
tst_dataset = DataSet(data_path=data_path, mode='tst')

def get_trn_loader(batch_size):
    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_loader

def get_val_loader(batch_size):
    train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader

def get_tst_loader(batch_size):
    train_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_loader