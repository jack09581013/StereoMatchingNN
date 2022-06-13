import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import utils
import os
import tools
import matplotlib.pyplot as plt
from utils.dataset import FlyingThings3D, random_subset
from torch.utils.data import DataLoader
from CSPN.cspn import CSPN
from GANet.GANet_small import GANetSmall
from GANet.GANet_small_deep import GANet_deep

height = 256
width = 512
max_disparity = 192
OVER_WRITE = True
FILE_PATH = '../../model/GANet_deep-one-image.torch'

# model = CSPN(max_disparity, 3, 12, 3).cuda()
# model = GANetSmall(max_disparity).cuda()
model = GANet_deep(max_disparity).cuda()

print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

if not OVER_WRITE and os.path.exists(FILE_PATH):
    print('Load ', FILE_PATH)
    model.load_state_dict(torch.load(FILE_PATH))

optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

train_dataset = FlyingThings3D((height, width), type='train', crop_seed=0)
train_dataset = random_subset(train_dataset, 1, seed=0)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

model.train()
for epoch in range(50):
    for batch_index, (X, Y) in enumerate(train_loader):
        # tools.tic()
        optimizer.zero_grad()

        if isinstance(model, GANet_deep):
            disp0, disp1, disp2, disp3 = model(X[:, 0:3, :, :], X[:, 3:6, :, :])
            loss0 = F.smooth_l1_loss(disp0, Y)
            loss1 = F.smooth_l1_loss(disp1, Y)
            loss2 = F.smooth_l1_loss(disp2, Y)
            loss3 = F.smooth_l1_loss(disp3, Y)
            loss = 0.2 * loss0 + 0.6 * loss1 + 0.8 * loss2 + loss3

        else:
            disp = model(X[:, 0:3, :, :], X[:, 3:6, :, :])
            loss = F.smooth_l1_loss(disp, Y)
            # raise Exception('Unknown model')

        loss.backward()
        optimizer.step()
        print('[{}] loss = {:.3f}'.format(epoch, loss))
        # tools.toc()
        # os.system('nvidia-smi')

model.eval()
for batch_index, (X, Y) in enumerate(train_loader):
    with torch.no_grad():
        cost, disp = model(X[:, 0:3, :, :], X[:, 3:6, :, :])
        loss = F.smooth_l1_loss(disp, Y)
        print('Test loss = {:.3f}'.format(loss))
        utils.plot_image_disparity(X[0], Y[0], disp[0], loss)

# [99] loss = 3.136
# Test loss = 5.943
torch.save(model.state_dict(), FILE_PATH)