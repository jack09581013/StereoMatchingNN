from GANet.module import *
import torch.nn as nn
import torch.optim as optim

# [SGA backward]
# cost max: 1.945687e+01
# cost_aggregation max: 1.040331e+01
# weight max: 7.707512e-01
# cost_grad max: nan
# weight_grad max: nan
# grad_output max: 1.422449e-01

# 0:00:00.949482
# 0:00:00.577739

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
y = torch.randn((batch, channels, max_disparity, height, width), device='cuda')
g = torch.randn((batch, 640, height, width), requires_grad=True, device="cuda")

model = SGA()

optimizer = optim.Adam([g], lr=0.01, betas=(0.9, 0.999))

for epoch in range(100):
    print('epoch = {}'.format(epoch))

    tools.tic()
    optimizer.zero_grad()
    cost = model(x, g)
    loss = F.smooth_l1_loss(cost, y)
    loss.backward()
    optimizer.step()

    print('loss = {:.3f}'.format(loss))
    tools.toc()

