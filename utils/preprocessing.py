# libraries

import os
import cv2
import glob
import numpy as np
import multiprocessing as mp

# functions


def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def scaleRadius(img, scale):
    """
    Part of Ben's technique
    https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/competitionreport.pdf
    """
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def aug_0(img_path):
    """
    Crop black area
    """

    img = cv2.imread(img_path)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def aug_1(img_path):
    """
    Ben's technique
    https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/competitionreport.pdf
    """

    scale = 300

    a = cv2.imread(img_path)
    a = scaleRadius(a, scale)
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
    b = np.zeros(a.shape)
    cv2.circle(
        b,
        (int(a.shape[1] / 2), int(a.shape[0] / 2)),
        int(scale * 0.9),
        (1, 1, 1),
        -1,
        8,
        0,
    )
    a = a * b - 128 * (1 - b)

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_1/{}".format(
            os.path.basename(img_path)
        ),
        a,
    )


def aug_2(img_path):
    """
    Ben's technique with median blur
    https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/competitionreport.pdf
    """

    scale = 300

    a = cv2.imread(img_path)
    a = scaleRadius(a, scale)

    k = np.min(a.shape[0:2]) // 20 * 2 + 1

    a = cv2.addWeighted(a, 4, cv2.medianBlur(a, k), -4, 128)
    b = np.zeros(a.shape)
    cv2.circle(
        b,
        (int(a.shape[1] / 2), int(a.shape[0] / 2)),
        int(scale * 0.9),
        (1, 1, 1),
        -1,
        8,
        0,
    )
    a = a * b - 128 * (1 - b)

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_2/{}".format(
            os.path.basename(img_path)
        ),
        a,
    )


def aug_3(img_path):
    """
    Cropping, apply CLAHE and then stack the green channel
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)
    image = image[:, :, 1]

    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    image = np.dstack((image, image, image))
    # all Green, RGB BGR doesn't matter
    return image


def aug_4(img_path):
    """
    Cropping, apply CLAHE, median blur and then stack the green channel
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    image = image[:, :, 1]
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    image = np.dstack((image, image, image))

    k = np.min(image.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(image, k)
    image = cv2.addWeighted(image, 4, bg, -4, 128)

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_4/{}".format(
            os.path.basename(img_path)
        ),
        image,
    )


def aug_5(img_path):
    """
    Cropping, apply CLAHE to all channels
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img



def aug_6(img_path):
    """
    Cropping, apply CLAHE to all channels with median blur
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    k = np.min(image.shape[0:2]) // 30 * 2 + 1
    bg = cv2.medianBlur(image, k)
    image = cv2.addWeighted(image, 4, bg, -4, 128)
    return image


def aug_7(img_path):
    """
    Cropping, dissect into 4, apply CLAHE, median blur and then stack the green channel
    Based on https://www.sciencedirect.com/science/article/pii/S2212017313005781
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    height, width, depth = image.shape

    image = image[:, :, 1]

    q1 = image[0 : int(height / 2), 0 : int(width / 2)]
    q2 = image[0 : int(height / 2), int(width / 2) :]
    q3 = image[int(height / 2) :, 0 : int(width / 2)]
    q4 = image[int(height / 2) :, int(width / 2) :]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    q1 = clahe.apply(q1)
    k = np.min(q1.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q1, k)
    q1 = cv2.addWeighted(q1, 4, bg, -4, 128)

    q2 = clahe.apply(q2)
    k = np.min(q2.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q2, k)
    q2 = cv2.addWeighted(q2, 4, bg, -4, 128)

    q3 = clahe.apply(q3)
    k = np.min(q3.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q3, k)
    q3 = cv2.addWeighted(q3, 4, bg, -4, 128)

    q4 = clahe.apply(q4)
    k = np.min(q4.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q4, k)
    q4 = cv2.addWeighted(q4, 4, bg, -4, 128)

    h1 = np.concatenate((q1, q2), axis=1)
    h2 = np.concatenate((q3, q4), axis=1)
    image = np.concatenate((h1, h2), axis=0)

    image = np.dstack((image, image, image))

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_7/{}".format(
            os.path.basename(img_path)
        ),
        image,
    )


def aug_8(img_path):
    """
    Cropping, dissect into 4, apply CLAHE, median blur across all channels
    Based on https://www.sciencedirect.com/science/article/pii/S2212017313005781
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    height, width, depth = image.shape

    q1 = image[0 : int(height / 2), 0 : int(width / 2)]
    q2 = image[0 : int(height / 2), int(width / 2) :]
    q3 = image[int(height / 2) :, 0 : int(width / 2)]
    q4 = image[int(height / 2) :, int(width / 2) :]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    q1 = cv2.cvtColor(q1, cv2.COLOR_BGR2LAB)
    q1[..., 0] = clahe.apply(q1[..., 0])
    q1 = cv2.cvtColor(q1, cv2.COLOR_LAB2BGR)
    k = np.min(q1.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q1, k)
    q1 = cv2.addWeighted(q1, 4, bg, -4, 128)

    q2 = cv2.cvtColor(q2, cv2.COLOR_BGR2LAB)
    q2[..., 0] = clahe.apply(q2[..., 0])
    q2 = cv2.cvtColor(q2, cv2.COLOR_LAB2BGR)
    k = np.min(q2.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q2, k)
    q2 = cv2.addWeighted(q2, 4, bg, -4, 128)

    q3 = cv2.cvtColor(q3, cv2.COLOR_BGR2LAB)
    q3[..., 0] = clahe.apply(q3[..., 0])
    q3 = cv2.cvtColor(q3, cv2.COLOR_LAB2BGR)
    k = np.min(q3.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q3, k)
    q3 = cv2.addWeighted(q3, 4, bg, -4, 128)

    q4 = cv2.cvtColor(q4, cv2.COLOR_BGR2LAB)
    q4[..., 0] = clahe.apply(q4[..., 0])
    q4 = cv2.cvtColor(q4, cv2.COLOR_LAB2BGR)
    k = np.min(q4.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(q4, k)
    q4 = cv2.addWeighted(q4, 4, bg, -4, 128)

    h1 = np.concatenate((q1, q2), axis=1)
    h2 = np.concatenate((q3, q4), axis=1)
    image = np.concatenate((h1, h2), axis=0)

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_8/{}".format(
            os.path.basename(img_path)
        ),
        image,
    )


def aug_9(img_path):
    """
    Cropping, apply CLAHE to all channels without converting to LAB
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image[..., 0] = clahe.apply(image[..., 0])
    image[..., 1] = clahe.apply(image[..., 1])
    image[..., 2] = clahe.apply(image[..., 2])

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_9/{}".format(
            os.path.basename(img_path)
        ),
        image,
    )


def aug_10(img_path):
    """
    Cropping, apply CLAHE to all channels without converting to LAB, then apply median
    """

    image = cv2.imread(img_path)
    image = crop_image_from_gray(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image[..., 0] = clahe.apply(image[..., 0])
    image[..., 1] = clahe.apply(image[..., 1])
    image[..., 2] = clahe.apply(image[..., 2])

    k = np.min(image.shape[0:2]) // 20 * 2 + 1
    bg = cv2.medianBlur(image, k)
    image = cv2.addWeighted(image, 4, bg, -4, 128)

    cv2.imwrite(
        "/media/ains/dec4dfd1-4e1b-4f58-8203-1e4a2fb67acf/preprocessing_experiment/aug_10/{}".format(
            os.path.basename(img_path)
        ),
        image,
    )


"""
images = glob.glob('/home/ains/PycharmProjects/aptos2019/data/aptos2019-blindness-detection/train_images/*.png')

pool = mp.Pool(mp.cpu_count())
pool.map(aug_2, images)
pool.close()
"""
