import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.optim as optim
import tools
import ganet_lib
import os
import numpy as np
from GANet.GANet import *

# from torch.utils.cpp_extension import load
# lib_cpp = load(name="lib_cpp", sources=["extensions/lib.cpp"], verbose=True)

height = 240
width = 576
max_disparity = 192

left_image = torch.randn((1, 3, height, width), requires_grad=False).cuda()
right_image = torch.randn((1, 3, height, width), requires_grad=False).cuda()

model = GANet(max_disparity).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for i in range(100):
    tools.tic()
    optimizer.zero_grad()
    output = model(left_image, right_image)
    # print('output min', output.view(-1).min())
    # print('output max', output.view(-1).max())
    # print('output.shape', output.shape)
    loss = (output - 0).pow(2).mean()
    loss.backward()
    # os.system('nvidia-smi')
    optimizer.step()

    print('loss {:.3f}'.format(loss))
    tools.toc()

# k1, k2, k3, k4 = torch.split(g, (x.size()[1] * 5, x.size()[1] * 5, x.size()[1] * 5, x.size()[1] * 5), 1)