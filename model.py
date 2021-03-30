# Written by Ken S. Zhang for ML Proj

import torch
import torch.nn as nn


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


class UpSampling(nn.Sequential):
    def __init__(self, inputSize, outputSize, scale=2):
        super(UpSampling, self).__init__()
        self.deConv_A = torch.nn.ConvTranspose2d(inputSize, outputSize, kernel_size=4, stride=2, padding=1, bias=True) 
        self.relu_A = nn.PReLU()

    def forward(self, x):
        output = self.relu_A(self.deConv_A(x))
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
        self.featEx_A = FeatureExtraction(inputSize=3, outputSize=32, ratio=2)
        self.Up_A = UpSampling(inputSize=32, outputSize=32)
        self.featEx_B = FeatureExtraction(inputSize=32, outputSize=64, ratio=2)
        self.Up_B = UpSampling(inputSize=64, outputSize=64)
        self.Distill = Refine(inputSize=64, midSizeList=[32, 8])

    def forward(self, x):
        x1 = self.featEx_A(x)
        x2 = self.Up_A(x1)
        x3 = self.featEx_B(x2)
        x4 = self.Up_B(x3)
        output = self.Distill(x4)
        return output


if __name__ == '__main__': 
    #model = FeatureExtraction(inputSize=3, outputSize=64, ratio=2)
    #model = UpSampling(inputSize=3, outputSize=64)
    #model = Refine(inputSize=64, midSizeList=[32, 8])
    model = Model()
    input = torch.rand(size=(1, 3, 384, 512)) 
    with torch.no_grad():
        out = model(input) 
        print(out.shape)