import os
import pdb

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

#import torch.utils.data as torchdata

from transforms import get_transforms


class AudioDataset(Dataset):
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
        self.fnames = self.df["recording_id"].values
        self.transform = get_transforms(phase, cfg)
        self.root = cfg.data_home / cfg.data_folder
        self.labels = df.values[:, 1:]

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = torch.Tensor(self.labels[idx].tolist())#.reshape(-1, 1)
        img = np.load(str(self.root / f'{fname}.npy'))
        img = img - img.min()
        img /= img.max()
        img = np.stack([img, img, img], axis=2)
        image = self.transform(image=img)["image"]
        return fname, image, label

    def __len__(self):
        total = len(self.df)
        #total = 100
        #total = int((total // self.batch_size) * self.batch_size)
        return total




class RainforestDataset(Dataset):
    def __init__(self, df, phase, cfg):
        self.filenames = []
        self.specs = []
        self.labels = []
        self.root = cfg.data_home / cfg.data_folder
        self.transform = get_transforms(phase, cfg)
        for idx, row in df.iterrows():
            filename = Path(row['file_name'])
            self.filenames.append(str(filename))
            label = int(row['label'])
            label_array = np.zeros(cfg.num_classes, dtype=np.single)
            label_array[label] = 1.0
            self.labels.append(label_array)

            img = Image.open(str(self.root / filename))
            mel_spec = np.array(img)
            img.close()
            mel_spec = mel_spec / 255
            mel_spec = np.stack((mel_spec, mel_spec, mel_spec), axis=2)

            self.specs.append(mel_spec)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, item):
        # Augment here if you want
        fname = self.filenames[item]
        img = self.specs[item]

        #import pdb; pdb.set_trace()
        img = self.transform(image=img)['image']
        return fname, img, self.labels[item]
