import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import os
import os.path
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
import numpy as np

from utils import download_url, check_integrity

class CIFAR10(data.Dataset):
    """ Prepares CIFAR10 Dataset.
    Args:
        root:           Root directory of dataset where directory 'cifar-10-batches-py' exists.
        split:          Split to be created (train, validation, test).
        train_ratio:    Percentage of training data to be used on train split.
        download:       Downloads the dataset from the internet and puts it in root directory.
        transform:      Optional transforms to apply on data.`
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [['data_batch_1', 'c99cafc152244af753f735de768cd75f'], ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'], ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'], ['data_batch_4', '634d18415352ddfa80567beed471001a'], ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],]
    test_list = [['test_batch', '40351d587109b95175f43aff81a1287e'],]
    meta = {'filename': 'batches.meta', 'key': 'label_names', 'md5': '5ff9c542aee3614f3951f8cda6e48888',}

    def __init__(self, root, split = 'train', train_ratio = 0.8, transform = None, download = False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split 
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found. Set download to true.')

        if self.split == 'test':
            data_list = self.test_list
        else:
            data_list = self.train_list

        self.data = []
        self.targets = []

        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

        if self.split != 'test':
            num_data = len(self.data)
            self.num_class = len(np.unique(self.targets))

            train_num = int(num_data * self.train_ratio)
            if self.split == 'train':
                self.data = self.data[:train_num]
                self.targets = self.targets[:train_num]
            elif self.split == 'val':
                self.data = self.data[train_num:num_data]
                self.targets = self.targets[train_num:num_data]
            else:
                self.data = self.data[num_data:]
                self.targets = self.targets[num_data:]
        else:
            num_data = len(self.data)
            self.num_class = len(np.unique(self.targets))
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]

    def get_data_labels(self):
        return self.targets

class CIFAR100(CIFAR10):
    """ Prepares CIFAR100 Dataset. """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d'],]
    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],]
    meta = {'filename': 'meta', 'key': 'fine_label_names', 'md5': '7973b15100ade9c7d40fb424638fde48',}
