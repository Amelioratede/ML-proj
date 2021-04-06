# Written by Ken S. Zhang for ML Proj

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from model import Model
from div2k import trainData, validationData
from utils import AvgMeter, ToNumpy, PSNR


LOG = open('./evallog.txt','a')
ckptDir = './ckpt'


def logPrint(string): 
    LOG.write(string + "\n") 
    print(string)


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-resolution <Evaluation>')
    parser.add_argument('-s', '--scale', default=4, type=int, help='Upsampling scale of super-resolution')
    parser.add_argument('-d', '--dir', default=None, type=str, help='Exact directory of ckpt path')  
    parser.add_argument('-c', '--ckpt', default=None, type=int, help='The selected checkpoint for evaluation')
    parser.add_argument('-r', '--reduction', default=3, type=int, help='Reduction of input size') 
    args = parser.parse_args()

    # Hyperparams   
    scale = args.scale
    selectDir = args.dir
    ckptPt = args.ckpt
    reduction = args.reduction

    # Time & Config
    currTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logPrint("\n-------- Time: {} --------".format(currTime))
    logPrint("Config: Scale-{} | Reduction-{} ".format(scale, reduction))
    
    # Load Model
    model = Model()
    if (selectDir == None):
        if (ckptPt == None):
            ckptLs = os.listdir('{}/'.format(ckptDir)) 
            if ".DS_Store" in ckptLs:
                ckptLs.remove(".DS_Store")
            ckptPt = len(ckptLs) - 1
        ckptLoadPath = os.path.join(ckptDir, 'checkpoint_{}'.format(ckptPt))
        logPrint('Model loaded from Epoch #{}.'.format(ckptPt))
    else:
        ckptLoadPath = selectDir
        logPrint('Model loaded from {}.'.format(ckptLoadPath))
    model.load_state_dict(torch.load(ckptLoadPath)) 

    # Load Data
    dataVal = validationData(scale=scale, reduction=reduction)
    logPrint('Data Loaded. Quantity of Validation Instances:{}.'.format(len(dataVal)))

    # Meters
    modelMeter = AvgMeter()
    baselineMeter = AvgMeter()
    lossMeter = AvgMeter()

    # Translator
    numpyTrans = ToNumpy()

    # Loss Function
    lossHandle = nn.L1Loss()

    logPrint('Evaluating...')
    model.eval()

    # Evaluation
    for _, sample in enumerate(dataVal):
        if (_%10==0): 
            print("\tIn Progress:{}/{}".format(_, len(dataVal)))

        # Prepare & Feed
        imgGt, imgInput = sample["groundTruth"], sample["input"] 
        with torch.no_grad():
            imgOutput = model(imgInput)
        
        # Compute Baseline
        imgBaseline = F.interpolate(imgInput, scale_factor=scale, mode='bicubic', align_corners=True)
        lossMeter.append(lossHandle(imgGt, imgBaseline))

        # Translate
        imgGt, imgOutput, imgBaseline = numpyTrans([imgGt, imgOutput, imgBaseline])

        # Compute Metrics
        psnrModel = PSNR(imgGt, imgOutput)
        psnrBaseline = PSNR(imgGt, imgBaseline)
        modelMeter.append(psnrModel)
        baselineMeter.append(psnrBaseline)
    
    # Evaluation Outcome 
    logPrint("Loss of the Baseline:{:.4f}".format(lossMeter.movingAvg()))  
    logPrint("Avg PSNR:\nModel vs.Baseline\n{:.4f}dB {:.4f}dB".format(modelMeter.movingAvg(), baselineMeter.movingAvg()))  


if __name__ == '__main__': 
    main()
