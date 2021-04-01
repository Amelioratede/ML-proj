# Written by Ken S. Zhang for ML Proj

import os
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils import data
import numpy as np
from utils import getValTransforms, getTrainTransforms


splits = ["valid", "train"]
DIV2kBasePath = '/home/shanyizhang/Documents/div2k'
# Data Available at https://data.vision.ee.ethz.ch/cvl/DIV2K/


class DIV2kBase(data.Dataset):
    def __init__(self, root, split):
        assert(split in splits)
        self.split = split
        self.splitRoot = os.path.join(root, 'DIV2K_{}_HR'.format(split))
        self.imgs = os.listdir(self.splitRoot)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index]


class DIV2k(DIV2kBase):
    def __init__(self, root, split, scale, reduction=1, transforms=None):
        super().__init__(root, split)
        self.scale = scale
        self.reduction = reduction
        self.transforms = transforms

    def __getitem__(self, index):
        imgName = super().__getitem__(index)
        imgPath = os.path.join(self.splitRoot, imgName)
        img = Image.open(imgPath)
        width = (img.size[0] // self.reduction // self.scale) * self.scale
        height = (img.size[1] // self.reduction // self.scale) * self.scale
        imgOriginal = img.resize((width, height), Image.BICUBIC)
        imgLow = imgOriginal.resize((width // self.scale, height // self.scale), Image.BICUBIC)
        del img
        sample = {'groundTruth': imgOriginal, 'input': imgLow}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


def validationData(scale, reduction):
    valLoader = DIV2k(root=DIV2kBasePath, split='valid', scale=scale, reduction=reduction, transforms=getValTransforms())
    return DataLoader(valLoader, batch_size=1, shuffle=False)


def trainData(scale, reduction):
    trainLoader = DIV2k(root=DIV2kBasePath, split='train', scale=scale, reduction=reduction, transforms=getTrainTransforms())
    return DataLoader(trainLoader, batch_size=1, shuffle=True)


if __name__ == '__main__':
    dset = DIV2k(root=DIV2kBasePath, split='train', scale=4, transforms=None)
    for i in range(len(dset)):
        instance = dset[i]
        gt = instance['groundTruth']
        low = instance['input']
        #gt.save("gt.png")
        #low.save("low.png")
