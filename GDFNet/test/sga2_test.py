from GANet.module import *
import torch.nn as nn
import torch.optim as optim
import os

batch = 1
channels = 32
height = 240
width = 576
max_disparity = 192
downsampling = 4

height //= downsampling
width //= downsampling
max_disparity //= downsampling

x = torch.randn((batch, channels, max_disparity, height, width), device='cuda')
y = torch.randn((batch, channels*4, max_disparity, height, width), device='cuda')
g = torch.randn((batch, 1280, height, width), requires_grad=True, device="cuda")

model = SGA2(3)

optimizer = optim.Adam([g], lr=1, betas=(0.9, 0.999))

for epoch in range(100):
    print('epoch = {}'.format(epoch))

    tools.tic()
    optimizer.zero_grad()
    cost = model(x, g)
    # os.system('nvidia-smi')
    loss = F.smooth_l1_loss(cost, y)
    loss.backward()
    optimizer.step()

    print('loss = {:.3f}'.format(loss))
    tools.toc()


