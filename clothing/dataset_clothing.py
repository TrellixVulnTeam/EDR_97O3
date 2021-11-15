import os
import torch
import torch.utils.data as data
import numpy as np
import PIL
from PIL import Image

class Clothing1M(data.Dataset):
    def __init__(self, root = None, split = 'train', transform = None):
        self.root = root
        self.transform = transform

        if split == 'train':
            file_path = os.path.join(self.root, 'noisy_train_key_list.txt')
            label_path = os.path.join(self.root, 'train_label.txt')
        elif split == 'val':
            file_path = os.path.join(self.root, 'clean_val_key_list.txt')
            label_path = os.path.join(self.root, 'val_label.txt')
        else:
            file_path = os.path.join(self.root, 'clean_test_key_list.txt')
            label_path = os.path.join(self.root, 'test_label.txt')

        with open(file_path) as fid:
            image_list = [line.strip() for line in fid.readlines()]

        with open(label_path) as fid:
            label_list = [int(line.strip()) for line in fid.readlines()]

        self.image_list = image_list
        self.label_list = label_list  
        self.targets = self.label_list

    def __getitem__(self, index):
        image_file_name = self.image_list[index]
        image_path = os.path.join(self.root, image_file_name)

        image = Image.open(image_path)
        image = image.resize((256, 256), resample = PIL.Image.BICUBIC)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        label = self.label_list[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.label_list)

    def update_corrupted_label(self, noise_label):
        self.label_list[:] = noise_label[:]
        self.targets = self.label_list

def get_data_labels():
    trainlist = 'data/noisy_train_key_list.txt'
    vallist = 'data/clean_val_key_list.txt'
    testlist = 'data/clean_test_key_list.txt'
    cleanlist = 'data/clean_label_kv.txt'
    noisylist = 'data/noisy_label_kv.txt'

    fid = open(trainlist)
    train_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(vallist)
    val_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(testlist)
    test_list = [line.strip() for line in fid.readlines()]
    fid.close()

    fid = open(cleanlist)
    clean_list = [line.strip().split(' ') for line in fid.readlines()]
    fid.close()

    fid = open(noisylist)
    noisy_list = [line.strip().split(' ') for line in fid.readlines()]
    fid.close()

    label_map = dict()
    for m in noisy_list:
        label_map[m[0]] = m[1]

    train_labels = []
    for t in train_list:
        label = label_map[t]
        train_labels.append(label)

    label_map = dict()
    for m in clean_list:
        label_map[m[0]] = m[1]

    val_labels = []
    for t in val_list:
        label = label_map[t]
        val_labels.append(label)

    test_labels = []
    for t in test_list:
        label = label_map[t]
        test_labels.append(label)

    with open('data/train_label.txt', 'w') as fid:
        for p in train_labels:
            fid.write('{}\n'.format(p))

    with open('data/val_label.txt', 'w') as fid:
        for p in val_labels:
            fid.write('{}\n'.format(p))

    with open('data/test_label.txt', 'w') as fid:
        for p in test_labels:
            fid.write('{}\n'.format(p))

    return True

if __name__ == '__main__':
    get_data_labels()
