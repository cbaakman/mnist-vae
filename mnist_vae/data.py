#!/usr/bin/env python

from typing import Tuple
import os
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):

    def __init__(self, path_prefix: str, num_samples: int, image_size: int):

        self.num_samples = num_samples
        self.image_size = image_size

        self.path_prefix = path_prefix

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        with open(os.path.join(self.path_prefix + '-images.idx3-ubyte'), 'rb') as f:
            f.seek(16, 0)
            f.seek(self.image_size * self.image_size * index, 1)

            image = torch.frombuffer(f.read(self.image_size * self.image_size), dtype=torch.uint8)
            image = image.reshape(1, self.image_size, self.image_size)

        with open(os.path.join(self.path_prefix + '-labels.idx1-ubyte'), 'rb') as f:
            f.seek(8, 0)
            f.seek(index, 1)

            label = torch.frombuffer(f.read(1), dtype=torch.uint8)

        return image, label

