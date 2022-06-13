import torch
import torch.nn as nn
import torch.nn.functional as F
import GANet.GANet_small as ga
import numpy as np
import os
import torch.optim as optim

class TestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

batch = 1
height = 240
width = 576
x = torch.ones((batch, 3, height, width), device='cuda')
y = torch.full((batch, 3, height, width), 3, device='cuda')

model = TestCNN().cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

for i in range(200):
    optimizer.zero_grad()
    out = model(x)
    loss = F.smooth_l1_loss(out, y)
    loss.backward()
    optimizer.step()
    print('{} loss = {:.3f}'.format(i, loss))



