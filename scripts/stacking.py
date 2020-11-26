#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdb
import os
import sys
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

sys.path.append("..")
from models import Model, get_model
from utils import *
from image_utils import *

# from submission import get_best_threshold


# In[2]:


class Dataset(data.Dataset):
    def __init__(self, root, df, phase, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.ext = ".png" if phase == "test" else ""
        self.TTA = albumentations.Compose(
            [
                # albumentations.RandomRotate90(p=1),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),
            ]
        )
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                # albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + self.ext)
        #         print(path)

        image = id_to_image(
            path,
            resize=True,
            size=self.size,
            augmentation=False,
            subtract_median=True,
            clahe_green=False,
        )

        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        # return 100
        return self.num_samples


def get_predictions(model, testset, tta):
    """return all predictions on testset in a list"""
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if tta:
            # images.shape [n, 3, 96, 96] where n is num of 1+tta
            for images in batch:
                preds = model(images.to(device))  # [n, num_classes]
                predictions.append(preds.mean(dim=0).detach().tolist())
        else:
            preds = model(batch[:, 0].to(device))
            preds = preds.detach().tolist()  # [1]
            predictions.extend(preds)

    return np.array(predictions)


def get_model_name_fold(model_folder_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/
    model_folder = model_folder_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2]  # fold0
    fold = fold.split("f")[-1]  # 0
    return model_name, int(fold)


# In[3]:


home = os.path.abspath(os.path.dirname(__file__))
ckpt_path_list = [
    "weights/168_efficientnet-b5_f0_poma/ckpt14.pth",
    "weights/168_efficientnet-b5_f1_poma/ckpt14.pth",
    "weights/168_efficientnet-b5_f2_poma/ckpt14.pth",
    "weights/168_efficientnet-b5_f3_poma/ckpt14.pth",
    "weights/168_efficientnet-b5_f4_poma/ckpt14.pth",
]
model_name = "efficientnet-b5"
folder = "168_efficientnet-b5_poma"

mkdir(os.path.join(home, "weights/ensemble", folder))
tta = 0
size = 300
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
use_cuda = True
num_classes = 1
num_workers = 12
batch_size = 16
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True
torch.set_default_tensor_type("torch.cuda.FloatTensor")


model = get_model(model_name, num_classes, pretrained=None)
model.to(device)
model.eval()

# ### train predictions

phase = "train"
df = pd.read_csv(os.path.join(home, "data/train.csv"))
train_labels = df["diagnosis"].values
root = os.path.join(home, "data/all_images/")
trainset = DataLoader(
    Dataset(root, df, phase, size, mean, std, tta),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if use_cuda else False,
)


# generate predictions using all models
all_predictions = []
for idx, ckpt in enumerate(ckpt_path_list):
    print("model: %s" % ckpt)
    ckpt_path = os.path.join(home, ckpt)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    predictions = get_predictions(model, trainset, tta)
    all_predictions.append(predictions)

train_path = os.path.join(home, "weights/ensemble", folder, "train.npy")
np.save(train_path, all_predictions)

# ### test predictions

print("Starting test predictions")
phase = "test"
df = pd.read_csv(os.path.join(home, "data/sample_submission.csv"))
root = os.path.join(home, "data/test_images/")
testset = DataLoader(
    Dataset(root, df, phase, size, mean, std, tta),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if use_cuda else False,
)


# generate predictions using all models
base_thresholds = np.array([0.5, 1.5, 2.5, 3.5])
all_predictions = []
for idx, ckpt in enumerate(ckpt_path_list):
    print("model: %s" % ckpt)
    ckpt_path = os.path.join(home, ckpt)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    predictions = get_predictions(model, testset, tta)
    preds = predict(predictions, base_thresholds)
    print(np.unique(preds, return_counts=True)[1])
    all_predictions.append(predictions)

test_path = os.path.join(home, "weights/ensemble", folder, "test.npy")
np.save(test_path, all_predictions)


# preds = predict(predictions, base_thresholds)
# print(np.unique(preds, return_counts=True)[1])
#
#
## In[29]:
#
#
##sub_path = 'weights/168_efficientnet_poma_ensemble.csv'
# df = pd.read_csv('../data/sample_submission.csv')
# df.loc[:, 'diagnosis'] = preds
# df.to_csv(sub_path, index=False)


# In[ ]:
