# ML-proj : Super-Image Super-Resolution

**Ken S. Zhang | Shera (Xirui) Li | Yiwen Wang**

## Dataset 

Data Available at https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Training

All the training is executed on **cuda5.cims.nyu.edu**.

```
python3 train.py
```

-------- Time: 04/20/2021, 03:14:00 --------

Config:

Scale-4 | Epochs-20 | Learning Rate-0.0001 | Batch Size-1

Reduction-3 | Halt-800

Use 1 GPU(s).

Model created. New start. Good Luck!

Data Loaded. Quantity of Training Instances:800.

Training at Epoch #0...

	Iteration:0 | Mov Avg Loss:0.6199
	
	Iteration:10 | Mov Avg Loss:0.2612
	
	Iteration:20 | Mov Avg Loss:0.2035

......

## Evaluation

-------- Time: 04/18/2021, 02:11:03 --------

Config: Scale-4 | Reduction-3 

Use 2 GPU(s).

Model loaded from Epoch #1.

Data Loaded. Quantity of Validation Instances:100.

Evaluating...

Loss of the Baseline:0.0433

Avg PSNR:

Model vs.Baseline

25.4708dB 23.5021dB


## Visualization

Config: Scale-4 | Reduction-2 

Model loaded from Epoch #0.

Data Loaded. Quantity of Visualization Instances:2.

Visualizing...

	nyush-1.jpeg - PSNR: 24.2418dB 21.7920dB (Model vs.Baseline) | Margin:2.4497dB
	
	nyush-2.jpeg - PSNR: 24.6449dB 22.4020dB (Model vs.Baseline) | Margin:2.2428dB

Avg PSNR:

Model vs.Baseline

24.4433dB 22.0970dB



## Overview of our Model

kenzhang@10-209-97-226 ML-proj-main % python3 model.py

Input Size: torch.Size([1, 3, 10, 20])

Output Size: torch.Size([1, 3, 40, 80])

​	 featEx_A.conv_A.weight 3456

​	 featEx_A.conv_A.bias 128

​	 featEx_A.relu_A.weight 1

​	 featEx_A.conv_B.weight 147456

​	 featEx_A.conv_B.bias 128

​	 Up_A.deConv_A.weight 524288

​	 Up_A.deConv_A.bias 256

​	 Up_A.relu_A.weight 1

​	 Up_A.conv_B.weight 589824

​	 Up_A.conv_B.bias 256

​	 Up_A.relu_B.weight 1

​	 Up_A.conv_C.weight 589824

​	 Up_A.conv_C.bias 256

​	 featEx_B.conv_A.weight 589824

​	 featEx_B.conv_A.bias 256

​	 featEx_B.relu_A.weight 1

​	 featEx_B.conv_B.weight 589824

​	 featEx_B.conv_B.bias 256

​	 Up_B.deConv_A.weight 2097152

​	 Up_B.deConv_A.bias 512

​	 Up_B.relu_A.weight 1

​	 Up_B.conv_B.weight 2359296

​	 Up_B.conv_B.bias 512

​	 Up_B.relu_B.weight 1

​	 Up_B.conv_C.weight 2359296

​	 Up_B.conv_C.bias 512

​	 Distill.featEx.conv_A.weight 2359296

​	 Distill.featEx.conv_A.bias 512

​	 Distill.featEx.relu_A.weight 1

​	 Distill.featEx.conv_B.weight 2359296

​	 Distill.featEx.conv_B.bias 512

​	 Distill.conv_A.weight 32768

​	 Distill.conv_A.bias 64

​	 Distill.relu_A.weight 1

​	 Distill.conv_B.weight 192

​	 Distill.conv_B.bias 3

Parameters In Total: 14605963
