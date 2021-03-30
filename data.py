# Written by Ken S. Zhang for ML Proj

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        imgGt, imgInput = sample['groundtruth'], sample['input']
        if random.random() < 0.5:
            imgGt = imgGt.transpose(Image.FLIP_LEFT_RIGHT)
            imgInput = imgInput.transpose(Image.FLIP_LEFT_RIGHT)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        imgGt, imgInput = sample['groundtruth'], sample['input']
        if random.random() < 0.5:
            imgGt = imgGt.transpose(Image.FLIP_TOP_BOTTOM)
            imgInput = imgInput.transpose(Image.FLIP_TOP_BOTTOM)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, image_low = sample['image'], sample['image_low']
        image = self.to_tensor(image)
        image_low = self.to_tensor(image_low)

        return {'image': image, 'image_low': image_low}

    def to_tensor(self, pic):
        img = torch.from_numpy(np.array(pic,dtype=np.float32))
        img = img.view(pic.size[1], pic.size[0], 3)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img.float().div(255)


def getNoTransform():
    return transforms.Compose([ToTensor(is_test=True)])

def getDefaultTrainTransform():
    return transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),ToTensor(is_test=False)])
