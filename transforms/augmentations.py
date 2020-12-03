import cv2
import random
from albumentations import (
    HorizontalFlip,
    IAAPerspective,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    RandomContrast,
    RandomBrightness,
    Flip,
    OneOf,
    Compose as ImgCompose,
    RandomGamma,
    ElasticTransform,
    ChannelShuffle,
    RGBShift,
    Rotate,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    CenterCrop,
    PadIfNeeded
)


from albumentations.pytorch import ToTensor
from albumentations.core.transforms_interface import ImageOnlyTransform
import librosa
import numpy as np
from pathlib import Path
import colorednoise as cn

class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)

class AddGaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_amplitude=0.5, **kwargs):
        super().__init__(always_apply, p)

        self.noise_amplitude = (0.0, max_noise_amplitude)

    def apply(self, y: np.ndarray, **params):
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_amplitude).astype(y.dtype)
        return augmented


class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented

class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_steps=5, sr=32000):
        super().__init__(always_apply, p)

        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented

class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1.2):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        super().__init__(always_apply, p)

        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented


class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented

def resize_sa(img, size):
    '''Resize, keeping the aspect ratio same
    Larger side is set to `size`, smaller side adjusted
    acc to aspect ratio
    image.shape -> height, width
    cv2.resize takes (width, height)
    '''

    h, w, _ = img.shape
    if h <= w: # equality is imp.
        nh, nw = int(size * (h/w) ), size
    elif h > w:
        nh, nw = size, int(size * (w/h) )
    nimg = cv2.resize(img, (nw, nh))
    return nimg


def strong_aug(p=1):
    """for reference, doesn't help"""
    return Compose(
        [
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
            OneOf(
                [
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            OneOf(
                [
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.3),
                ],
                p=0.2,
            ),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomContrast(),
                    RandomBrightness(),
                ],
                p=0.3,
            ),
            # HueSaturationValue(p=0.3),
        ],
        p=p,
    )



class MyCenterCrop(ImageOnlyTransform):
    """Customized center crop, randomly center crops, keeping the aspect ratio
    intact.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0):
        super(MyCenterCrop, self).__init__(always_apply, p)

    def apply(self, image, **params):
        w, h, _ = image.shape
        ratio = random.choice([0.8, 0.85, 0.9, 0.95])
        aug = CenterCrop(int(w * ratio), int(h * ratio), p=1)
        image = aug(image=image)['image']
        return image

    def get_transform_init_args_names(self):
        return ()



class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

def get_transforms(phase, cfg):
    size = cfg["size"]
    mean = eval(cfg["mean"])
    std = eval(cfg["std"])

    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                #Transpose(p=0.5),
                #HorizontalFlip(p=0.5),
                #ShiftScaleRotate(
                #    shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=180, p=0.7
                #),
                #RandomBrightnessContrast(0.1, 0.1, p=0.2),
                #strong_aug()

            ]
        )
    list_transforms.extend(
        [
            Resize(size[0], size[1]),
            #PadIfNeeded(size, size, p=1, border_mode=1),

            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(normalize=None),  # [6]
            #ToTensor(),  # [6]
        ]
    )
    return ImgCompose(list_transforms)



def get_audio_transforms(phase, cfg):
    sr = cfg.sr
    if phase == "train":
        transforms = Compose([
            OneOf([
            GaussianNoiseSNR(min_snr=10),
            PinkNoiseSNR(min_snr=10)
            ]),
            #PitchShift(max_steps=2, sr=sr),
            #TimeStretch(),
            TimeShift(sr=sr),
            VolumeControl(mode="sine")
        ])
    else:
        transforms = Compose([])
    return transforms
