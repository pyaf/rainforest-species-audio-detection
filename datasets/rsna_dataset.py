import os
import pdb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transforms import get_transforms
from .preprocessing import get_image


class RSNADataset(Dataset):
    """training dataset."""

    def __init__(self, df, phase, cfg):
        """
        Args:
            df: the dataframe
            phase: train/val
            cfg: config file

        """

        self.phase = phase
        self.df = df
        self.size = cfg.size
        self.batch_size = cfg.batch_size[phase]
        self.num_samples = self.df.shape[0]
        self.fnames = self.df["filename"].values
        self.transform = get_transforms(phase, cfg)
        self.root = cfg.data_home / cfg.data_folder
        self.labels = df.values[:, 1:]
        #pdb.set_trace()
        #self.labels = []
        #for idx in range(self.num_samples):
        #    self.labels.append(df.iloc[idx].values[1:])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = torch.Tensor(self.labels[idx].tolist())#.reshape(-1, 1)
        image = get_image(self.root / f'{fname}.dcm')
        #pdb.set_trace()
        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        total = len(self.df)
        #total = 100
        #total = int((total // self.batch_size) * self.batch_size)
        return total


