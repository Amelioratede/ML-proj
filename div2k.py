# Written by Ken S. Zhang for ML Proj

import os
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torch.utils import data


splits = ["valid", "train"]
DIV2kBasePath = '/Users/kenzhang/Downloads'
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
    def __init__(self, root, split, scale, transforms=None):
        super().__init__(root, split)
        self.transforms = transforms
        self.scale = scale

    def __getitem__(self, index):
        imgName = super().__getitem__(index)
        imgPath = os.path.join(self.splitRoot, imgName)
        img = Image.open(imgPath)
        width = (img.size[0] // self.scale) * self.scale
        height = (img.size[1] // self.scale) * self.scale
        imgOriginal = img.resize((width, height), Image.BICUBIC)
        imgLow = imgOriginal.resize((width // self.scale, height // self.scale), Image.BICUBIC)
        sample = {'groundTruth': imgOriginal, 'input': imgLow}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


def validationData(batchSize, scale):
    valLoader = DIV2k(root=DIV2kBasePath, split='valid', scale=scale, transforms=getDefaultTrainTransform())
    return DataLoader(valLoader, batchSize, shuffle=True)


def trainData(batchSize, scale):
    trainLoader = DIV2k(root=DIV2kBasePath, split='train', scale=scale, transforms=None)
    return DataLoader(trainLoader, batchSize, shuffle=False)


if __name__ == '__main__':
    dset = DIV2k(root=DIV2kBasePath, split='train', scale=4, transforms=None)
    for i in range(len(dset)):
        instance = dset[i]
        gt = instance['groundTruth']
        low = instance['input']
        #gt.save("gt.png")
        #low.save("low.png")
