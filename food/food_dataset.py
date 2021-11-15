import os
from os import listdir
import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image
import numpy as np

def resize(img, size, max_size = 1000):
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        sw = sh = float(size) / size_min
        
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BICUBIC)

class Food101N(Dataset):
    """ Prepares Food101N Dataset.
    Args:
        root:           Root directory of dataset.
        split:          Split to be created (train, validation, test).
        transform:      Optional transforms to apply on data.`
    """
    def __init__(self, split = 'train', root = None, transform = None):
        if split == 'train':
            self.image_list = np.load(os.path.join(root, 'train_images.npy'))
            self.targets = np.load(os.path.join(root, 'train_targets.npy'))
        elif split == 'val':
            self.image_list = np.load(os.path.join(root, 'valid_images.npy'))
            self.targets = np.load(os.path.join(root, 'valid_targets.npy'))
        else:
            self.image_list = np.load(os.path.join(root, 'test_images.npy'))
            self.targets = np.load(os.path.join(root, 'test_targets.npy'))

        self.targets = self.targets - 1
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image = resize(image, 256)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]

def gen_data_list():

    classlist = 'data/Food-101N_release/meta/classes.txt'
    imagelist = 'data/Food-101N_release/meta/imagelist.tsv'
    vallist = 'data/Food-101N_release/meta/verified_train.tsv'
    filepath = 'data/Food-101N_release/images'

    classmap = dict()
    with open(classlist) as fp:
        for i, line in enumerate(fp):
            row = line.strip()
            classmap[row] = i
    num_class = len(classmap)
    print('Num Classes: ', num_class)

    targets = []
    imgs = []
    with open(imagelist) as fp:
        fp.readline()
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(classmap[class_name])
            imgs.append(os.path.join(filepath, line.strip()))

    targets = np.array(targets)
    imgs = np.array(imgs)
    print('Num Train Images: ', len(imgs))
    np.save('data/train_images.npy', imgs)
    np.save('data/train_targets.npy', targets)

    targets = []
    imgs = []
    with open(vallist) as fp:
        fp.readline()
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            valid = row[1][-1]
            if valid:
                targets.append(classmap[class_name])
                imgs.append(os.path.join(filepath, line.strip().split('\t')[0]))

    targets = np.array(targets)
    imgs = np.array(imgs)
    print('Num Valid Images: ', len(imgs))
    np.save('data/valid_images.npy', imgs)
    np.save('data/valid_targets.npy', targets)

    testlist = 'data/food-101/meta/test.txt'
    filepath = 'data/food-101/images'
    targets = []
    imgs = []
    with open(testlist) as fp:
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(classmap[class_name])
            imgs.append(os.path.join(filepath, line.strip() + '.jpg'))
            
    targets = np.array(targets)
    imgs = np.array(imgs)
    print('Num Test Images: ', len(imgs))
    np.save('data/test_images.npy', imgs)
    np.save('data/test_targets.npy', targets)
    
    return True

if __name__ == '__main__':
    gen_data_list()
