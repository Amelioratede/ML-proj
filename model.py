# Written by Ken S. Zhang for ML Proj

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtraction(nn.Sequential):
    def __init__(self, inputSize, outputSize, ratio):
        super(FeatureExtraction, self).__init__()
        self.ratio = ratio
        self.midSize = outputSize//ratio
        self.conv_A = nn.Conv2d(inputSize, self.midSize, kernel_size=3, stride=1, padding=1)
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(inputSize+self.midSize, outputSize, kernel_size=3, stride=1, padding=1)
        self.relu_B = nn.PReLU()

    def forward(self, x):
        phase_A = self.relu_A(self.conv_A(x))
        concat = torch.cat([phase_A, x], dim=1)
        output = self.relu_B(self.conv_B(concat))
        return output


class BottleNeck(nn.Sequential):
    def __init__(self, inputSize, outputSize):
        super(BottleNeck, self).__init__()
        self.conv_A = nn.Conv2d(inputSize, outputSize, kernel_size=1, stride=1, padding=0)
        self.relu_A = nn.PReLU()

    def forward(self, x):
        output = self.relu_A(self.conv_A(x))
        return output


class UpSampling(nn.Sequential):
    def __init__(self, inputSize, outputSize, scale=2):
        super(UpSampling, self).__init__()
        self.scale = scale
        self.deConv_A = torch.nn.ConvTranspose2d(inputSize, outputSize, kernel_size=4, stride=2, padding=1, bias=True) 
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(outputSize, outputSize, kernel_size=3, stride=1, padding=1) 
        self.relu_B = nn.PReLU()

    def forward(self, x):
        phase_A = self.relu_A(self.deConv_A(x))
        output = self.relu_B(self.conv_B(phase_A))
        return output


class Interpolation(nn.Sequential):
    def __init__(self, inputSize, outputSize, scale=2):
        super(Interpolation, self).__init__()
        self.scale = scale
        self.conv_A = nn.Conv2d(inputSize, outputSize, kernel_size=3, stride=1, padding=1) 
        self.relu_A = nn.PReLU()

    def forward(self, x):
        phase_A = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=True)
        output = self.relu_A(self.conv_A(phase_A))
        return output


class Refine(nn.Sequential):
    def __init__(self, inputSize, midSizeList):
        super(Refine, self).__init__()
        assert(len(midSizeList)==2)
        self.conv_A = nn.Conv2d(inputSize, midSizeList[0]//2, kernel_size=3, stride=1, padding=1)
        self.bnConv_A = nn.Conv2d(inputSize, midSizeList[0]//2, kernel_size=1, stride=1, padding=0)
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(midSizeList[0], midSizeList[1]//2, kernel_size=3, stride=1, padding=1)
        self.bnConv_B = nn.Conv2d(midSizeList[0], midSizeList[1]//2, kernel_size=1, stride=1, padding=0)
        self.relu_B = nn.PReLU()
        self.bnConv_C = nn.Conv2d(midSizeList[1], 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        phase_A_1 = self.conv_A(x)
        phase_A_2 = self.bnConv_A(x)
        concat = torch.cat([phase_A_1, phase_A_2], dim=1)
        phase_A = self.relu_A(concat)
        phase_B_1 = self.conv_B(phase_A)
        phase_B_2 = self.bnConv_B(phase_A)
        concat = torch.cat([phase_B_1, phase_B_2], dim=1)
        phase_B = self.relu_B(concat)
        output = self.bnConv_C(phase_B)
        return output


class Model(nn.Module): 
    def __init__(self):
        super(Model, self).__init__() 
        self.featEx_A = FeatureExtraction(inputSize=3, outputSize=64, ratio=2)
        self.Up_A = UpSampling(inputSize=64, outputSize=128)
        self.featEx_B = FeatureExtraction(inputSize=128, outputSize=128, ratio=2)
        self.Up_B = UpSampling(inputSize=128, outputSize=128)
        self.Distill = Refine(inputSize=128, midSizeList=[64, 16])

    def forward(self, x):
        x1 = self.featEx_A(x)
        x2 = self.Up_A(x1)
        x3 = self.featEx_B(x2)
        x4 = self.Up_B(x3)
        output = self.Distill(x4)
        return output


class ModelPlus(nn.Module): 
    def __init__(self):
        super(ModelPlus, self).__init__() 
        # UP
        self.featEx_A = FeatureExtraction(inputSize=3, outputSize=64, ratio=2)
        self.Interp_A = Interpolation(inputSize=64, outputSize=128)
        self.Ups_A = UpSampling(inputSize=64, outputSize=128)
        self.BottleN_A = BottleNeck(inputSize=128*2, outputSize=128)
        # UP
        self.featEx_B = FeatureExtraction(inputSize=128, outputSize=128, ratio=2)
        self.Interp_B = Interpolation(inputSize=128, outputSize=256)
        self.Ups_B = UpSampling(inputSize=128, outputSize=256)
        self.BottleN_B = BottleNeck(inputSize=256*2, outputSize=256)

        self.Distill = Refine(inputSize=256, midSizeList=[128, 64])

    def forward(self, x):
        phase_A = self.featEx_A(x)
        phase_B_1 = self.Interp_A(phase_A)
        phase_B_2 = self.Ups_A(phase_A)
        concat = torch.cat([phase_B_1, phase_B_2], dim=1)
        phase_C = self.BottleN_A(concat)
        phase_D = self.featEx_B(phase_C)
        phase_E_1 = self.Interp_B(phase_D)
        phase_E_2 = self.Ups_B(phase_D)       
        concat = torch.cat([phase_E_1, phase_E_2], dim=1)
        phase_F = self.BottleN_B(concat)
        output = self.Distill(phase_F)
        return output



if __name__ == '__main__': 
    #model = FeatureExtraction(inputSize=3, outputSize=64, ratio=2)
    #model = UpSampling(inputSize=3, outputSize=64)
    #model = Refine(inputSize=64, midSizeList=[32, 8])
    model = ModelPlus()
    input = torch.rand(size=(1, 3, 10, 20)) 
    with torch.no_grad():
        out = model(input) 
        print(out.shape)