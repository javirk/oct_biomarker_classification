from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
import logging
import random
import skimage.transform
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


if __name__ == '__main__':
    #transform = transforms.Compose([Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    d = OCTHDF5Dataset('../../../Datasets/ambulatorium_all_slices.hdf5')
    d._get_posweights()
