import numpy as np
from pathlib import Path


def min_max_norm(img, zeroone=True):
    preped = ((img-img.min()) / (img.max()-img.min()))
    if zeroone:
        return preped
    else:
        return preped*2-1


def standardization(img):
    return (img-img.mean()) / img.std()


def percentile_thresholding(img, percentile=1):
    min_percentile = np.percentile(img, percentile)
    max_percentile = np.percentile(img, 100-percentile)

    img[img >= max_percentile] = max_percentile
    img[img <= min_percentile] = min_percentile
    return img


def get_subjects(data_dir, test_indexes=None, valid_indexes=None):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if test_indexes is None:
        test_indexes = [i for i in range(10)]

    if valid_indexes is None:
        valid_indexes = [i for i in range(10, 20)]

    subjs = list(data_dir.glob("*"))
    test_subjs = list(map(subjs.__getitem__, test_indexes))
    valid_subjs = list(map(subjs.__getitem__, valid_indexes))
    train_subjs = list(set(subjs) - set(test_subjs) - set(valid_subjs))
    return train_subjs, valid_subjs, test_subjs


def get_pair(subjs):
    images, masks = [], []
    for subj in subjs:
        fs = list(subj.glob("*"))
        mask = list(subj.glob("*mask*"))
        masks.extend(mask)
        images.extend(list(set(fs) - set(mask)))

    images = list(map(str, images))
    images.sort(key=lambda f: (f.split("_")[-2], f.split("_")[-1].split(".")[0]))

    masks = list(map(str, masks))
    masks.sort(key=lambda f: (f.split("_")[-3], f.split("_")[-2].split(".")[0]))
    return images, masks
