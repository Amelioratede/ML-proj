# Written by Ken S. Zhang for ML Proj

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from model import Model
from utils import AvgMeter, ToNumpy, PSNR, visualData, numpyWriter, imgWriter, refine


LOG = open('./visuallog.txt','a')
ckptDir = './ckpt'


def logPrint(string): 
    LOG.write(string + "\n") 
    print(string)


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-resolution <Visualization>')
    parser.add_argument('-s', '--scale', default=4, type=int, help='Upsampling scale of super-resolution')
    parser.add_argument('-d', '--dir', default=None, type=str, help='Exact directory of ckpt path')  
    parser.add_argument('-c', '--ckpt', default=None, type=int, help='The selected checkpoint for evaluation')
    parser.add_argument('-v', '--visual', default="./visual", type=str, help='Directory of input images') 
    parser.add_argument('-r', '--reduction', default=2, type=int, help='Reduction of input size') 
    args = parser.parse_args()

    # Hyperparams   
    scale = args.scale
    selectDir = args.dir
    ckptPt = args.ckpt
    reduction = args.reduction
    visualDir = args.visual

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
    dataVisual = visualData(root=visualDir, scale=scale, reduction=reduction)
    logPrint('Data Loaded. Quantity of Visualization Instances:{}.'.format(len(dataVisual)))

    # Meters
    modelMeter = AvgMeter()
    baselineMeter = AvgMeter()

    # Translator
    numpyTrans = ToNumpy()

    # Loss Function
    lossHandle = nn.L1Loss()

    logPrint('Visualizing...')
    model.eval()

    # Evaluation
    for _, sample in enumerate(dataVisual):

        # Prepare & Feed
        imgGt, imgInput, imgName = sample["groundTruth"], sample["input"],  sample["imgName"][0]
        with torch.no_grad():
            imgOutput = model(imgInput)
        
        # Compute Baseline
        imgBaseline = F.interpolate(imgInput, scale_factor=scale, mode='bicubic', align_corners=True)

        # Translate
        imgGt, imgOutput, imgBaseline = numpyTrans([imgGt, imgOutput, imgBaseline])

        # Compute Metrics
        psnrModel = PSNR(imgGt, imgOutput)
        psnrBaseline = PSNR(imgGt, imgBaseline)
        margin = psnrModel - psnrBaseline
        modelMeter.append(psnrModel)
        baselineMeter.append(psnrBaseline)

        # Instance Outcome
        logPrint("\t{} - PSNR: {:.4f}dB {:.4f}dB (Model vs.Baseline) | Margin:{:.4f}dB".format(imgName, psnrModel, psnrBaseline, margin))

        # Store Images
        imgWriter("{}_output".format(visualDir), imgName.split('.')[0], imgGt, imgOutput, imgBaseline)
    
    # Evaluation Outcome 
    logPrint("Avg PSNR:\nModel vs.Baseline\n{:.4f}dB {:.4f}dB".format(modelMeter.movingAvg(), baselineMeter.movingAvg()))  


if __name__ == '__main__': 
    main()
