# Written by Ken S. Zhang for ML Proj

import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import torch.nn.functional as F
from datetime import datetime
from model import Model


torch.autograd.set_detect_anomaly(True) 
LOG = open('./log.txt','a')
ckptDir = './ckpt'
os.makedirs(ckptDir, exist_ok=True)


def logPrint(string): 
    LOG.write(string + "\n") 
    print(string)


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-resolution')
    parser.add_argument('-s', '--scale', default=4, type=int, help='Upsampling scale of super-resolution') 
    parser.add_argument('-e', '--epochs', default=40, type=int, help='Number of epochs to run') 
    parser.add_argument('-l', '--learningrate', default=0.0001, type=float, help='Initial learning rate') 
    parser.add_argument('-b', '--batchsize', default=4, type=int, help='Batch size') 
    args = parser.parse_args()

    # Hyperparams 
    scale = args.scale
    epochs = args.epochs
    learningRate = args.learningrate
    batchSize = args.batchsize

    # Time & Config
    currTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logPrint("Time: {}".format(currTime))
    logPrint("Config: Scale:{} | Epoch:{} | Learning Rate:{} | Batch Size:{}".format(scale, epochs, learningRate, batchSize))

    # Load model
    ckptLs = os.listdir('{}/'.format(ckptDir)) 
    if (len(ckptLs) > 0):
        currentEpoch = len(ckptLs)
        model = Model()
        model.load_state_dict(torch.load('{}/checkpoint_{}'.format(ckptDir, currentEpoch-1))) 
        logPrint('Model loaded. Begin from Epoch #{}.'.format(currentEpoch))
    else:
        model = Model()
        currentEpoch = 0
        logPrint('Model created. New start.')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), learningRate) 

    # Loss Function
    loss = nn.L1Loss()
