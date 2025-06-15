import os
import numpy as np
import json

import torch
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from util.cate import cate


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target


class MRARDataset(Dataset):
    def __init__(self, folder, files='Imagenet2012_mrar_files.txt', loader=default_loader,
                 transform=None):
        self.cate = cate
        self.folder = folder
        self.loader = loader
        self.transform = transform
        self.files = []
        with open(files, 'r') as r:
            for i in r:
                tmp = json.loads(i)
                cate_name, files = tmp['cate'], tmp['files']
                self.files.append([self.cate[cate_name], cate_name, files])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print(idx)
        cate_id, cate_name, files = self.files[idx]

        if self.transform is not None:
            sample0 = self.transform(self.loader(self.folder + '/' + cate_name + '/' + files[0]))
            sample1 = self.transform(self.loader(self.folder + '/' + cate_name + '/' + files[1]))
            sample2 = self.transform(self.loader(self.folder + '/' + cate_name + '/' + files[2]))
            sample3 = self.transform(self.loader(self.folder + '/' + cate_name + '/' + files[3]))

        return (sample0, sample1, sample2, sample3), int(cate_id)
