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
    Compose,
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


from albumentations.torch import ToTensor
from albumentations.core.transforms_interface import ImageOnlyTransform


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


def get_transforms(phase, cfg):
    size = cfg["size"]
    mean = eval(cfg["mean"])
    std = eval(cfg["std"])

    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                Transpose(p=0.5),
                Flip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=180, p=0.7
                ),
                RandomBrightnessContrast(0.1, 0.1, p=0.2),
                #strong_aug()

            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            #PadIfNeeded(size, size, p=1, border_mode=1),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(normalize=None),  # [6]
        ]
    )
    return Compose(list_transforms)


