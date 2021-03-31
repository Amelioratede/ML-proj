# Written by Ken S. Zhang for ML Proj

import argparse
import os
import torch
import torch.nn as nn
from datetime import datetime
from model import Model
from div2k import trainData, validationData
from utils import AvgMeter, lrDecay


torch.autograd.set_detect_anomaly(True) 
LOG = open('./trainlog.txt','a')
ckptDir = './ckpt'
os.makedirs(ckptDir, exist_ok=True)
logTimeInterval = 10


def logPrint(string): 
    LOG.write(string + "\n") 
    print(string)


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-resolution <Train>')
    parser.add_argument('-s', '--scale', default=4, type=int, help='Upsampling scale of super-resolution') 
    parser.add_argument('-e', '--epochs', default=40, type=int, help='Total number of epochs to run') 
    parser.add_argument('-l', '--learningrate', default=0.0001, type=float, help='Initial learning rate') 
    parser.add_argument('-b', '--batchsize', default=1, type=int, help='Batch size') 
    args = parser.parse_args()

    # Hyperparams 
    scale = args.scale
    epochs = args.epochs
    learningRate = args.learningrate
    batchSize = args.batchsize

    # Time & Config
    currTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logPrint("\n-------- Tensor Flys at {} --------".format(currTime))
    logPrint("Config: Scale-{} | Epochs-{} | Learning Rate-{} | Batch Size-{}".format(scale, epochs, learningRate, batchSize))

    # Load Model
    model = Model()
    ckptLs = os.listdir('{}/'.format(ckptDir)) 
    if (len(ckptLs) > 0):
        currentEpoch = len(ckptLs)
        ckptLoadPath = os.path.join(ckptDir, 'checkpoint_{}'.format(currentEpoch-1))
        model.load_state_dict(torch.load(ckptLoadPath)) 
        logPrint('Model loaded. Begin from Epoch #{}.'.format(currentEpoch))
    else:
        currentEpoch = 0
        logPrint('Model created. New start. Good Luck!')

    # Loss Function
    lossHandle = nn.L1Loss()

    # Meters
    trainLossMeter = AvgMeter()
    valLossMeter = AvgMeter()

    # Load Data
    dataTrain = trainData(scale=scale, batchSize=batchSize)
    dataVal = validationData(scale=scale)
    logPrint('Data Loaded. Quantity of Training Instances:{}.'.format(len(dataTrain)))

    for e in range(currentEpoch, epochs): 
        logPrint('Training at Epoch #{}...'.format(e))
        model.train()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lrDecay(learningRate, e))

        # Training
        for i, sample in enumerate(dataTrain): 
            if (i == len(dataTrain)//2):
                break
            # Prepare & Feed
            imgGt, imgInput = sample["groundTruth"], sample["input"]
            optimizer.zero_grad()
            imgGt = torch.autograd.Variable(imgGt)
            imgInput = torch.autograd.Variable(imgInput) 
            imgOutput = model(imgInput)
            loss = lossHandle(imgOutput, imgGt)

            # Back Propagation
            loss.backward()
            optimizer.step()
            trainLossMeter.append(loss)

            # Monitor
            if (i % logTimeInterval == 0):
                logPrint("\tIteration:{} | Mov Avg Loss:{:.4f}".format(i, trainLossMeter.movingAvg()))

        logPrint("Training Loss of the epoch:{:.4f}".format(trainLossMeter.movingAvg()))

        # Save Model      
        ckptLoadPath = os.path.join(ckptDir, 'checkpoint_{}'.format(e))
        torch.save(model.state_dict(), ckptLoadPath)
        logPrint('Model saved.')

        # Evaluation  
        logPrint('Evaluating at Epoch #{}...'.format(e))
        model.eval()
        for i, sample in enumerate(dataVal):
            imgGt, imgInput = sample["groundTruth"], sample["input"] 
            with torch.no_grad():
                imgOutput = model(imgInput)
                valLossMeter.append(lossHandle(imgOutput, imgGt))
                
        logPrint("Evaluation Loss of the epoch:{:.4f}".format(valLossMeter.movingAvg()))  
        logPrint("----------")  

        # Renew Meters
        trainLossMeter.flush()
        valLossMeter.flush()


if __name__ == '__main__': 
    main()