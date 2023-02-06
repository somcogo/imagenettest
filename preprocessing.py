from PIL import Image
import h5py
from glob import glob
import os
import numpy as np
import time

t1 = time.time()

trn_mean = [122.4626755957, 114.2584061279, 101.3746757056]
trn_std = [70.6315337607, 68.611443708, 71.9308860917]

val_mean = [123.0029179199, 114.6230518799, 101.5222020996]
val_std = [70.6385718769, 68.675766869, 72.150312154]

tst_mean = [120.3040514404, 112.595478125, 99.9852074219]
tst_std = [70.2993703366, 68.3517029374, 71.3411933592]

train_data_path = 'data/train'
val_data_path = 'data/val/images'
test_data_path = 'data/test/images'

label_dict = dict()
label_path_list = glob(os.path.join(train_data_path, '*'))
label_path_list.sort()
for i, path in enumerate(label_path_list):
    label = os.path.split(path)[-1]
    label_dict[label] = i

train_path_list = glob(os.path.join(train_data_path, '*', 'images', '*'))
train_path_list.sort()
val_path_list = glob(os.path.join(val_data_path, '*'))
val_path_list.sort()
tst_path_list = glob(os.path.join(test_data_path, '*'))
tst_path_list.sort()

trn_images = np.zeros((100000, 64, 64, 3), dtype=np.float32)
val_images = np.zeros((10000, 64, 64, 3), dtype=np.float32)
tst_images = np.zeros((10000, 64, 64, 3), dtype=np.float32)

for i, path in enumerate(train_path_list):
    image = Image.open(path)
    data = np.asarray(image)
    if len(data.shape) == 2:
        data = np.stack([data, data, data], axis=2)
    trn_images[i] = data

trn_images = np.divide(trn_images - trn_mean, trn_std)
trn_lables_strings = [os.path.split(path)[-1][:9] for path in train_path_list]
trn_lables = [label_dict[label] for label in trn_lables_strings]

t2 = time.time()
print('loaded train data', t2 - t1)

with open('data/val/val_annotations.txt') as f:
    lines = f.readlines()
    val_tuples = [line.split("\t")[:2] for line in lines]

for i, tuple in enumerate(val_tuples):
    path = os.path.join(val_data_path, tuple[0])
    image = Image.open(path)
    data = np.asarray(image)
    if len(data.shape) == 2:
        data = np.stack([data, data, data], axis=2)
    val_images[i] = data

val_images = np.divide(val_images - val_mean, val_std)
val_labels_strings = [tuple[1] for tuple in val_tuples]
val_labels = [label_dict[label] for label in val_labels_strings]

t3 = time.time()
print('loaded val data', t3 - t2)

for i, path in enumerate(tst_path_list):
    image = Image.open(path)
    data = np.asarray(image)
    if len(data.shape) == 2:
        data = np.stack([data, data, data], axis=2)
    tst_images[i] = data

tst_images = np.divide(tst_images - tst_mean, tst_std)
t4 = time.time()
print('loaded test data', t4 - t3)

f_trn = h5py.File('data/tiny_imagenet_trn.hdf5', 'w')
f_trn.create_dataset('data', data=trn_images, chunks=(1, 64, 64, 3), dtype='float32')
f_trn.create_dataset('labels', data=trn_lables)
f_trn.close()

t5 = time.time()
print('train file', t5 - t4)

f_val = h5py.File('data/tiny_imagenet_val.hdf5', 'w')
f_val.create_dataset('data', data=val_images, chunks=(1, 64, 64, 3), dtype='float32')
f_val.create_dataset('labels', data=val_labels)
f_val.close()

t6 = time.time()
print('val file', t6 - t5)

f_tst = h5py.File('data/tiny_imagenet_tst.hdf5', 'w')
f_tst.create_dataset('data', data=tst_images, chunks=(1, 64, 64, 3), dtype='float32')
f_tst.close()

t7 = time.time()
print('test file', t7 - t6)

