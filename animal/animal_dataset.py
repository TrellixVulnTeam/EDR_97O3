import os
from os import listdir
import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image
import numpy as np

class Animal10(Dataset):
    """ Prepares Animal10 Dataset.
    Args:
        root:           Root directory of dataset.
        split:          Split to be created (train, validation, test).
        transform:      Optional transforms to apply on data.`
    """
    def __init__(self, root = None, split = 'train', transform = None):
        self.image_dir = os.path.join(root, split)
        self.image_files = [f for f in listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        self.targets = []

        for path in self.image_files:
            label = path.split('_')[0]
            self.targets.append(int(label))

        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]
