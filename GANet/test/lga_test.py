from GANet.module import *
import torch.nn as nn
import torch.optim as optim

batch = 1
height = 240
width = 576
max_disparity = 192

x = torch.ones((batch, 3, height, width), device='cuda')
y = torch.full((batch, 3, height, width), 3, device='cuda')
g = torch.randn((batch, 75, height, width), requires_grad=True, device='cuda')

# lg1: 75, H, W
model = LGA(5)

optimizer = optim.Adam([g], lr=0.01, betas=(0.9, 0.999))

for i in range(500):
    # tools.tic()
    optimizer.zero_grad()
    cost = model(x, g)
    loss = (cost - y).pow(2).mean()
    loss.backward()
    optimizer.step()

    # print('x mean = {:e}'.format(x.mean()))
    # print('g mean = {:e}'.format(g.mean()))
    print('{} loss = {:.3f}'.format(i, loss))
    # tools.toc()

