from __future__ import print_function, division
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
import h5py
import logging
import random
import skimage.transform
from skimage import io
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


class OCTHDF5Dataset(Dataset):
    """OCT HDF5 dataset."""

    def __init__(self, hdf5_file, image_set="data/slices", slice_set="meta/slice_pos", label_set='data/markers',
                 transform_image=None):
        '''
        labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
        '''
        self.dataset = None
        self.quantity = 0
        self.hdf5_file = hdf5_file
        self.image_set_name = image_set
        self.slice_set_name = slice_set
        self.label_set_name = label_set

        self.image_set = h5py.File(self.hdf5_file, 'r')[self.image_set_name]
        self.label_set = h5py.File(self.hdf5_file, 'r')[self.label_set_name]
        self.slice_set = h5py.File(self.hdf5_file, 'r')[self.slice_set_name] if slice_set is not None else None

        with h5py.File(self.hdf5_file, 'r') as file:
            self.dataset_len = file[image_set].shape[0]

        self.transform_image = transform_image

        self.weights = self._get_weights_loss()
        self.posweights = self._get_posweights()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.image_set is None:
            self.image_set = h5py.File(self.hdf5_file, 'r')[self.image_set_name]

        if self.label_set is None:
            self.label_set = h5py.File(self.hdf5_file, 'r')[self.label_set_name]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_set[idx] / 256
        bscan_number = self.slice_set[idx] if self.slice_set_name is not None else 24
        label = self.label_set[idx].astype(np.float32)
        image = np.concatenate([image, image, image], axis=-1)
        center = torch.tensor([24])

        seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.transform_image:
            random.seed(seed)
            image = self.transform_image(image)

        sample = {'images': image, 'slices': bscan_number, 'labels': label, 'center': center}
        return sample

    def _get_weights_loss(self):
        labels_sum = np.sum(self.label_set, axis=0)
        largest_class = max(labels_sum)
        weights = largest_class / labels_sum
        weights = torch.from_numpy(weights)
        return weights

    def _get_posweights(self):
        class_counts = np.sum(self.label_set, axis=0)
        pos_weights = np.ones_like(class_counts)
        neg_counts = [len(self.label_set) - pos_count for pos_count in class_counts]
        for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)

        return torch.tensor(pos_weights.astype(int), dtype=torch.float)


class OCTSlicesDataset(Dataset):
    """OCT Slices dataset."""

    def __init__(self, dtype, csv_paths, slices_path, target='majority', transform_image=None):
        '''
        labels: ['Cannot Grade', 'Double Layer Sign', 'Drusen', 'Drusenoid PED',
                 'Healthy', 'Not in the list', 'PED Serous', 'Reticular Drusen',
                 'Subretinal Fibrosis', 'cRORA', 'iRORA']
        '''

        self.dataset_type = dtype

        # Do not consider the slices where target is NaN (majority not obtained or slices not allocated to grader)
        dfs = []
        for csv_path in csv_paths:
            dfs.append(pd.read_csv(csv_path))
        df = pd.concat(dfs, ignore_index=True)

        df = df[['filename', 'volume_filename', 'dataset', 'slice_number', 'biomarker', target]]
        df = df[df.dataset == self.dataset_type]
        df = df.dropna(axis=0, subset=target)

        # slice_number not used for the time being
        self.df = df.pivot(index=['filename', 'volume_filename'], columns='biomarker', values=target)
        self.image_set = self.df.index.get_level_values(0)
        self.label_set = self.df.values.astype(int)
        self.volume_set = self.df.index.get_level_values(1)

        self.slices_path = slices_path

        # self.dataset = None
        # self.quantity = 0

        self.dataset_len = len(self.image_set)

        self.transform_image = transform_image

        self.weights = self._get_weights_loss()
        self.posweights = self._get_posweights()

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        if value.lower() in ['train', 'test']:
            self._dataset_type = value.title()
        else:
            raise ValueError('Dataset type must be either Train or Test (case insensitive)')

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.image_set is None:
            self.image_set = self.df.index.get_level_values(0)

        if self.label_set is None:
            self.label_set = self.df.values.astype(int)

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.slices_path.joinpath(self.image_set[idx])) / 256
        # volume_file = self.slice_set[idx]
        label = self.label_set[idx].astype(np.float32)
        image = np.stack([image, image, image], axis=-1)
        # center = torch.tensor([24])

        seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.transform_image:
            random.seed(seed)
            image = self.transform_image(image)

        sample = {'images': image, 'labels': label, 'paths': self.image_set[idx]} #, 'center': center}
        return sample

    def _get_weights_loss(self):
        labels_sum = np.sum(self.label_set, axis=0)
        largest_class = max(labels_sum)
        weights = largest_class / labels_sum
        weights = torch.from_numpy(weights)
        return weights

    def _get_posweights(self):
        class_counts = np.sum(self.label_set, axis=0)
        pos_weights = np.ones_like(class_counts)
        neg_counts = [len(self.label_set) - pos_count for pos_count in class_counts]
        for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)

        return torch.tensor(pos_weights.astype(int), dtype=torch.float)


if __name__ == '__main__':
    #transform = transforms.Compose([Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_path = Path(__file__).parents[1].joinpath('inputs')

    d = OCTSlicesDataset('train', csv_path=data_path.joinpath('annotation_10_percent_export.csv'), slices_path=data_path.joinpath('slices'))
    d._get_posweights()
