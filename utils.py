# Written by Ken S. Zhang for ML Proj

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import random
import os
import numpy as np
import math
import imageio


decayFactor = 0.8
maxPixelValue = 255.0


# -----------------------------------
# Utilies regarding Data Manipulation 
# -----------------------------------

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        imgGt, imgInput = sample['groundTruth'], sample['input']
        if random.random() < 0.5:
            imgGt = imgGt.transpose(Image.FLIP_LEFT_RIGHT)
            imgInput = imgInput.transpose(Image.FLIP_LEFT_RIGHT)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        imgGt, imgInput = sample['groundTruth'], sample['input']
        if random.random() < 0.5:
            imgGt = imgGt.transpose(Image.FLIP_TOP_BOTTOM)
            imgInput = imgInput.transpose(Image.FLIP_TOP_BOTTOM)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample


class RandomRotate(object):
    def __call__(self, sample):
        imgGt, imgInput = sample['groundTruth'], sample['input']
        if random.random() < 0.5:
            imgGt = imgGt.rotate(90, expand=True)
            imgInput = imgInput.rotate(90, expand=True)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample


class RandomZoomInOut(object):
    def __init__(self, scale):
        self.scale =  scale

    def __call__(self, sample):
        imgGt, imgInput = sample['groundTruth'], sample['input']
        randScalar = 1 + random.random()
        width = int(imgGt.size[0] / randScalar / self.scale) * self.scale
        height = int(imgGt.size[1] / randScalar / self.scale) * self.scale
        imgGt = imgGt.resize((width, height), Image.BICUBIC)
        imgInput = imgInput.resize((width // self.scale, height // self.scale), Image.BICUBIC)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample


class ToTensor(object):
    def __init__(self, testFlag):
        self.testFlag = testFlag

    def __call__(self, sample):
        imgGt, imgInput = sample['groundTruth'], sample['input']
        imgGt = self.to_tensor(imgGt)
        imgInput = self.to_tensor(imgInput)
        sample = {'groundTruth': imgGt, 'input': imgInput}
        return sample

    def to_tensor(self, pic):
        img = torch.from_numpy(np.array(pic, dtype=np.float32))
        img = img.view(pic.size[1], pic.size[0], 3)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.div(maxPixelValue)
        return img


class ToNumpy(object):
    def __init__(self, testFlag=True):
        self.testFlag = testFlag

    def __call__(self, sampleList):
        return (self.to_numpy(_) for _ in sampleList)

    def to_numpy(self, pic):
        img = torch.squeeze(pic).transpose(0, 2).transpose(0, 1)
        img = img.numpy() * maxPixelValue
        return img


def getNoTransforms():
    return transforms.Compose([ToTensor(testFlag=True)])


def getTrainTransforms(scale):
    return transforms.Compose([RandomRotate(), RandomHorizontalFlip(), RandomVerticalFlip(), RandomZoomInOut(scale=scale), ToTensor(testFlag=False)])


# -----------------------------------
# Auxiliary Utilies 
# -----------------------------------

class AvgMeter(object):
    def __init__(self):
        self.dataList = list()

    def append(self, item):
        self.dataList.append(item)
    
    def movingAvg(self):
        avg = sum(self.dataList) / len(self.dataList)
        return avg

    def flush(self):
        self.dataList = list()


# -----------------------------------
# Utilies regarding Evaluation Process
# -----------------------------------

def PSNR(left, right):
    MSE = np.mean((left - right) ** 2)
    result = 10 * math.log10(maxPixelValue**2/MSE)
    return result 


# -----------------------------------
# Utilies regarding Training Process
# -----------------------------------

def lrDecay(lr, epoch):
    lrGround = lr / 10.0
    lr = lr*(decayFactor**epoch)
    return max(lr, lrGround)


# -----------------------------------
# Utilies regarding Data Manipulation 
# -----------------------------------

class VisualDataBase(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(self.root)
        if ".DS_Store" in self.imgs:
            self.imgs.remove(".DS_Store")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index]


class VisualData(VisualDataBase):
    def __init__(self, root, scale, reduction=1, transforms=None):
        super().__init__(root)
        self.scale = scale
        self.reduction = reduction
        self.transforms = transforms

    def __getitem__(self, index):
        imgName = super().__getitem__(index)
        imgPath = os.path.join(self.root, imgName)
        img = Image.open(imgPath)
        width = int(img.size[0] / self.reduction / self.scale) * self.scale
        height = int(img.size[1] / self.reduction / self.scale) * self.scale
        imgOriginal = img.resize((width, height), Image.BICUBIC)
        imgLow = imgOriginal.resize((width // self.scale, height // self.scale), Image.BICUBIC)
        del img
        sample = {'groundTruth': imgOriginal, 'input': imgLow}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample['imgName'] = imgName
        return sample


def visualData(root, scale, reduction):
    visualLoader = VisualData(root=root, scale=scale, reduction=reduction, transforms=getNoTransforms())
    return DataLoader(visualLoader, batch_size=1, shuffle=False)


def refine(output):
    return np.uint8(np.clip(output, 0, 255))
    

def numpyWriter(root, imgName, gt, pred, baseline):
    imgDir = os.path.join(root, imgName)
    os.makedirs(imgDir, exist_ok=True)
    np.save(os.path.join(imgDir, "{}_gt.npy".format(imgName)), gt)
    np.save(os.path.join(imgDir, "{}_pred.npy".format(imgName)), pred)
    np.save(os.path.join(imgDir, "{}_baseline.npy".format(imgName)), baseline)


def imgWriter(root, imgName, gt, pred, baseline):
    imgDir = os.path.join(root, imgName)
    os.makedirs(imgDir, exist_ok=True)
    imgGt = Image.fromarray(np.uint8(gt))
    imgGt.save(os.path.join(imgDir, "{}_gt.jpeg".format(imgName)))
    imgPred = Image.fromarray(np.uint8(pred))
    imgPred.save(os.path.join(imgDir, "{}_pred.jpeg".format(imgName)))
    imgBaseline = Image.fromarray(np.uint8(baseline))
    imgBaseline.save(os.path.join(imgDir, "{}_baseline.jpeg".format(imgName)))
    imgs = [imgGt, imgBaseline, imgPred]
    imageio.mimsave(os.path.join(imgDir, "{}_combined.gif".format(imgName)), imgs, format='GIF', duration=1)