import cv2
import numpy as np


def load_image(path, size):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    return image


def load_ben_color(path, size, sigmaX=10, crop=False):
    """if crop=True: center crop retina"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop:
        image = crop_image(image)
    image = cv2.resize(image, (size, size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def crop_image(img, tol=10):
    """remove the black area, input should be BGR, for 3 channels"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0] // 2
    midline = im[cy, :]
    midline = np.where(midline > midline.mean() / 3)[0]
    if len(midline) > im.shape[1] // 2:
        x_start, x_end = np.min(midline), np.max(midline)
    else:  # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1] // 10, 9 * im.shape[1] // 10
    cx = (x_start + x_end) / 2
    r = (x_end - x_start) / 2
    return cx, cy, r


def resize_image(im, IMAGE_SIZE, augmentation=True):
    # Crops, resizes and potentially augments the image to IMAGE_SIZE
    cx, cy, r = info_image(im)
    scaling = IMAGE_SIZE / (2 * r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand() - 0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx, cy), rotation, scaling)
    M[0, 2] -= cx - IMAGE_SIZE / 2
    M[1, 2] -= cy - IMAGE_SIZE / 2
    return cv2.warpAffine(
        im, M, (IMAGE_SIZE, IMAGE_SIZE)
    )  # This is the most important line


def subtract_median_bg_image(im):
    k = np.max(im.shape) // 20 * 2 + 1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def subtract_gaussian_bg_image(im):
    k = np.max(im.shape) / 10
    bg = cv2.GaussianBlur(im, (0, 0), k)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def toCLAHEgreen(img):
    clipLimit = 2.0
    tileGridSize = (8, 8)
    img = np.array(img)
    green_channel = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cla = clahe.apply(green_channel)
    cla = clahe.apply(cla)
    cla = np.expand_dims(cla, -1)
    cla = np.repeat(cla, 3, -1)
    return cla


def id_to_image(
    path,
    resize=True,
    size=456,
    augmentation=False,
    subtract_gaussian=False,
    subtract_median=False,
    clahe=False,
    clahe_green=False,
):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if resize_image:
        im = resize_image(im, size, augmentation)
    if subtract_gaussian:
        im = subtract_gaussian_bg_image(im)
    if subtract_median:
        im = subtract_median_bg_image(im)
    if clahe_green:
        im = toCLAHEgreen(im)
    return im


def PP1(path, size):
    """
    PP1: preprocessing method 1
    read image, crop black area, bgr-lab -> apply CLAHE, lab-rbg, apply median filter
    """
    try:
        bgr = cv2.imread(path)
        bgr = crop_image(bgr)
        bgr = cv2.resize(bgr, (size, size))
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # Median filtering
        rgb = subtract_median_bg_image(rgb)
    except Exception as e:
        print(path)
        rgb = np.zeros([256, 256, 3])
    return rgb


def PP2(path):
    """
    read image, crop black area, apply CLAHE to each channel, apply median filter
    """
    bgr = cv2.imread(path)
    bgr = crop_image(bgr)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bgr[..., 0] = clahe.apply(bgr[..., 0])
    bgr[..., 1] = clahe.apply(bgr[..., 1])
    bgr[..., 2] = clahe.apply(bgr[..., 2])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Median filtering
    rgb = subtract_median_bg_image(rgb)
    return rgb
