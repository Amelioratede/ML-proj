# Written by Ken S. Zhang for ML Proj

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractionFirst(nn.Sequential):
    def __init__(self, inputSize, outputSize):
        super(FeatureExtractionFirst, self).__init__()
        self.conv_A = nn.Conv2d(inputSize, outputSize, kernel_size=3, stride=1, padding=1)
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(outputSize, outputSize, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.conv_B(self.relu_A(self.conv_A(x)))
        return output


class FeatureExtraction(nn.Sequential):
    def __init__(self, inputSize):
        super(FeatureExtraction, self).__init__()
        self.conv_A = nn.Conv2d(inputSize, inputSize, kernel_size=3, stride=1, padding=1)
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(inputSize, inputSize, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        phase_A = self.conv_B(self.relu_A(self.conv_A(x)))
        output = torch.add(phase_A, x)
        return output


class UpSampling(nn.Sequential):
    def __init__(self, inputSize, outputSize, scale=2):
        super(UpSampling, self).__init__()
        self.scale = scale
        self.deConv_A = torch.nn.ConvTranspose2d(inputSize, outputSize, kernel_size=4, stride=2, padding=1, bias=True) 
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(outputSize, outputSize, kernel_size=3, stride=1, padding=1) 
        self.relu_B = nn.PReLU()
        self.conv_C = nn.Conv2d(outputSize, outputSize, kernel_size=3, stride=1, padding=1) 

    def forward(self, x):
        phase_A = self.relu_A(self.deConv_A(x))
        phase_B = self.conv_C(self.relu_B(self.conv_B(phase_A)))
        output = torch.add(phase_A, phase_B)
        return output


class Refine(nn.Sequential):
    def __init__(self, inputSize, midSize):
        super(Refine, self).__init__()
        self.featEx = FeatureExtraction(inputSize)
        self.conv_A = nn.Conv2d(inputSize, midSize, kernel_size=1, stride=1, padding=0)
        self.relu_A = nn.PReLU()
        self.conv_B = nn.Conv2d(midSize, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        phase_A = self.featEx(x)
        output = self.conv_B(self.relu_A(self.conv_A(phase_A)))
        return output


class Model(nn.Module): 
    def __init__(self):
        super(Model, self).__init__() 
        self.featEx_A = FeatureExtractionFirst(inputSize=3, outputSize=64)
        self.Up_A = UpSampling(inputSize=64, outputSize=128)
        self.featEx_B = FeatureExtraction(inputSize=128)
        self.Up_B = UpSampling(inputSize=128, outputSize=256)
        self.Distill = Refine(inputSize=256, midSize=64)

    def forward(self, x):
        phase_A = self.featEx_A(x)
        phase_B = self.Up_A(phase_A)
        phase_C  = self.featEx_B(phase_B)
        phase_D = self.Up_B(phase_C)
        output = self.Distill(phase_D)
        return output


if __name__ == '__main__': 
    model = Model()
    input = torch.rand(size=(1, 3, 10, 20)) 
    with torch.no_grad():
        out = model(input) 
        print(out.shape)