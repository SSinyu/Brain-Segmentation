import cv2
import numpy as np
from math import ceil
from tensorflow.keras import utils
from albumentations import Compose, ElasticTransform, GridDistortion, CLAHE

from src.utils import percentile_thresholding, min_max_norm, get_subjects, get_pair

missing_subj = []

class DataLoader(utils.Sequence):
    def __init__(self, mode, data_path, batch_size, input_size=None, **kwargs):
        super(DataLoader, self).__init__()
        assert mode in ["train", "valid", "test"]

        self.mode = mode
        train_subjs, valid_subjs, test_subjs = get_subjects(data_path)

        if mode == "train":
            subjs = train_subjs
        elif mode == "valid":
            subjs = valid_subjs
        else:
            subjs = test_subjs

        self.images, self.masks = get_pair(subjs)
        self.indexes = np.arange(len(self.images))

        self.batch_size = batch_size

        if input_size is None:
            tmp_input = cv2.imread(self.images[0])
            self.input_size = tmp_input.shape
        else:
            self.input_size = input_size

        self.on_epoch_end()
        self.set_params()
        self.get_augmentation()

    def __len__(self):
        return ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size : (idx+1)*self.batch_size]
        return self.__getbatch__(indexes)

    def set_params(self, grid_distort=0., elastic_deform=0., histeq=0.):
        self.prob_distort = grid_distort
        self.prob_elastic = elastic_deform
        self.prob_histeq = histeq

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    @staticmethod
    def center_crop(img, crop_size):
        x, y, _ = img.shape
        startx = x//2 - (crop_size//2)
        starty = y//2 - (crop_size//2)
        return img[startx:(startx+crop_size), starty:(starty+crop_size), :]

    @staticmethod
    def pad_and_random_crop(img, mask, crop_size):
        assert img.shape == mask.shape
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h, w, _ = img.shape
        pad_size = (20,20)

        img = np.pad(img, (pad_size, pad_size,(0,0)), "reflect")
        mask = np.pad(mask, (pad_size, pad_size,(0,0)), "reflect")

        x = np.random.randint(0, pad_size[0])
        y = np.random.randint(0, pad_size[0])

        cropped_img = img[x:(x+crop_size[0]), y:(y+crop_size[1]), :]
        cropped_mask = mask[x:(x+crop_size[0]), y:(y+crop_size[1]), :]
        return cropped_img, cropped_mask

    @staticmethod
    def prep(img, p=1):
        img = percentile_thresholding(img, p)
        img = min_max_norm(img, False)
        return img

    def get_augmentation(self):
        self.aug = Compose([
            GridDistortion(num_steps=3, p=self.prob_distort),
            ElasticTransform(p=self.prob_elastic),
            CLAHE(p=self.prob_histeq)
        ])

    def __getbatch__(self, indexes):
        bx = np.zeros((self.batch_size, *self.input_size), dtype=np.float32)
        by = np.zeros((self.batch_size, *self.input_size), dtype=np.float32)

        for bi, i in enumerate(indexes):
            image = cv2.imread(self.images[i])
            mask = cv2.imread(self.masks[i])
            if self.mode == "train":
                image, mask = self.pad_and_random_crop(image, mask, self.input_size[:2])
                mask //= 225
                data = {"image":image, "mask":mask}
                aug = self.aug(**data)
                image, mask = aug["image"], aug["mask"]
            else:
                image = self.prep(image)
                mask //= 225

            bx[bi], by[bi] = image, mask
        return bx, by[...,0][...,np.newaxis]
