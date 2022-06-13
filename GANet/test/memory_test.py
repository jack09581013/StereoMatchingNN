import torch
import torch.nn.functional as F
import GANet.GANet_small as ga
import numpy as np
import os

batch = 1
height = 240
width = 576
x = torch.randn((batch, 3, height, width)).cuda()

model = ga.Feature().cuda()
y = model(x)
os.system('nvidia-smi')



