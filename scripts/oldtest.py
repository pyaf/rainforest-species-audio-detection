import pdb
import os
import cv2
import time
from glob import glob
import torch
import scipy
import pandas as pd
import numpy as np
from PIL import Image
import jpeg4py as jpeg
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
from models import Model, get_model
from augmentations import *
from utils import *
from image_utils import *
from preprocessing import *


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--model_folder_path",
        dest="model_folder_path",
        help="relative path to the folder where model checkpoints are saved",
    )
    parser.add_argument(
        "-e",
        "--epoch_range",
        nargs="+",
        type=int,
        dest="epoch_range",
        help="Epoch to start from",
    )  # usage: -e 10 20

    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="test",
    )
    parser.add_argument(
        "-s", "--size", dest="size", help="image size to use", default=256
    )

    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, predict_on, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.ext = ".png" if predict_on == "test" else ""
        """
        self.ext = ".tif" if predict_on == "train_mess" else ".png"
        print("Loading images...")
        if predict_on =="train":
            self.images = np.load('data/all_train_bgcc456.npy')
        elif predict_on == "test":
            self.images = np.load('data/all_test_bgcc456.npy')
        else:
            print('line 66 err')
        self.images = []  # because small dataset.
        for fname in tqdm(self.fnames):
            path = os.path.join(self.root, fname + self.ext)
            image = load_ben_color(path, size=self.size, crop=True)
            self.images.append(image)
        """
        print("Done")
        self.TTA = albumentations.Compose(
            [
                MyCenterCrop(p=0.2),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=120,
                    p=0.5,
                    #border_mode=cv2.BORDER_CONSTANT,
                ),
                albumentations.RandomBrightnessContrast(p=0.25),
            ]
        )
        self.transform = albumentations.Compose(
            [
                albumentations.Resize(size, size),
                albumentations.Normalize(mean=mean, std=std, p=1),
                AT.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # path = os.path.join(self.root, fname + self.ext)
        # image = load_ben_color(path, size=self.size, crop=True)
        # image = self.images[idx]
        # path = os.path.join(self.root, fname + '.npy')
        # image = np.load(path)
        path = os.path.join(self.root, fname + self.ext)
        # print(path)
        # image = id_to_image(path,
        #        resize=True,
        #        size=self.size,
        #        augmentation=False,
        #        subtract_median=True,
        #        clahe_green=True)
        # image = PP1(path)
        _, ext = os.path.splitext(path)
        if ext == ".jpeg" or ext == ".jpg":
           image = jpeg.JPEG(path).decode()
        else:
           image = Image.open(path)
           image = np.array(image)
        #image = aug_6(path)

        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta):  # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
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


if __name__ == "__main__":
    """
    Generates predictions on train/test set using the ckpts saved in the model folder path and saves them in npy_folder in npy format which can be analyses later for different thresholds
    """
    parser = get_parser()
    args = parser.parse_args()
    model_folder_path = args.model_folder_path
    predict_on = args.predict_on
    start_epoch, end_epoch = args.epoch_range
    model_name, fold = get_model_name_fold(model_folder_path)
    model_name = "efficientnet-b5"
    if predict_on == "test":
        sample_submission_path = "data/sample_submission.csv"
    elif predict_on == "train":
        sample_submission_path = "data/train.csv"
    elif predict_on == "train_mess":
        sample_submission_path = "data/train_messidor.csv"

    tta = 0  # number of augs in tta
    root = "data/test_images/"
    if predict_on == "train":
        root = "data/all_images/"
    elif predict_on == "train_mess":
        root = "external_data/messidor/train_images/"
    root = "data/aug_6"
    size = int(args.size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    use_cuda = True
    num_classes = 1
    num_workers = 8
    batch_size = 8
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    best_sub = "weights/submission812.csv"
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, size, mean, std, predict_on, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(model_name, num_classes, pretrained=None)
    model.to(device)
    model.eval()

    npy_folder = os.path.join(model_folder_path, f"{predict_on}_npy/{size}")
    mkdir(npy_folder)

    print(f"\nUsing model: {model_name} | fold: {fold}")
    print(f"Predicting on: {predict_on} set")
    print(f"Root: {root}")
    print(f"size: {size}")
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f"Saving predictions at: {npy_folder}")
    print(f"From epoch {start_epoch} to {end_epoch}")
    print(f"Using tta: {tta}\n")

    base_thresholds = np.array([0.5, 1.5, 2.5, 3.5])

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"Using ckpt{epoch}.pth")
        ckpt_path = os.path.join(model_folder_path, "ckpt%d.pth" % epoch)
        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        preds = get_predictions(model, testset, tta)
        best_thresholds = state["best_thresholds"]
        pred2 = predict(preds, base_thresholds)
        print("base:", np.unique(pred2, return_counts=True)[1])
        """
        print(f"Best thresholds: {best_thresholds}")
        pred1 = predict(preds, best_thresholds)
        print("best:", np.unique(pred1, return_counts=True)[1])
        """
        mat_to_save = [preds, best_thresholds]
        np.save(os.path.join(npy_folder, f"{predict_on}_ckpt{epoch}.npy"), mat_to_save)
        print("Predictions saved!")


"""
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
"""
