import torch
import torch.nn as nn
import torch.nn.functional as F
import ganet_lib
import numpy as np
import utils
from torch.utils.data import DataLoader, random_split, Subset
from CSPN.cspn import CSPN
from GANet.GANet_small import GANetSmall
import os

def l1_normalize():
    x = torch.randn((3, 3)).cuda()
    y = x.clone()
    print(x)
    x = F.normalize(x, p=1, dim=0)
    # ganet_lib.cuda_test(x)
    print(x)
    print(x[:, 0].abs().sum())

    for i in range(3):
        d = y[:, i].abs().sum()
        y[:, i] /= d

    print(y)

def train_test_split(full_dataset):
    x = torch.arange(0, 9).view(3, 3)
    print(x[0])

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    print('Train test size:', (train_size, test_size))

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

def epe_loss():
    x = torch.ones((4, 3))
    loss = utils.EPE_loss(x, 0)
    print(loss)

def deconv():
    # size = stride * (x - 1) + k - 2*p
    x = torch.ones((1, 1, 5))
    w = torch.ones((1, 1, 4))
    y = F.conv_transpose1d(x, w, stride=2, padding=1)
    print(y[0, 0])

def test_interpolate():
    x = torch.arange(0, 9, dtype=torch.float).view(1, 1, 3, 3)
    print(x)
    y = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    print(y)
    y = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    print(y)

def test_pad_memory():
    x = torch.ones((2, 32, 1000, 1000), dtype=torch.float)
    li = []
    for i in range(10):
        os.system('nvidia-smi')
        t = F.pad(x, [1, 1, 1, 1])
        li.append(t)

def test_probability_volume():
    x = torch.zeros((2, 3, 5, 5), dtype=torch.float)
    y = torch.zeros((2, 5, 5), dtype=torch.float)

    y[0, 0, 0] = 0.8
    y = y.unsqueeze(1)
    index = y.long()
    mid = y - index

    assert torch.all(y < 3 - 1), f'disparity must lower than max disparity {3 - 1}'

    print(index)
    x.scatter_(1, index, mid)
    x.scatter_(1, index + 1, 1 - mid)
    print(x[0, :, 0, 0].view(-1))

    # x = torch.zeros((5, 5), dtype=torch.float)
    # index = torch.zeros((5, 1), dtype=torch.long)
    # index[0] = 3
    # index[1] = 2
    #
    # print(index)
    # x.scatter_(1, index, 1)
    # print(x)

def test_cross_entropy():
    y = torch.randn((10,), dtype=torch.float)
    t = torch.zeros((10,), dtype=torch.float)

    y = F.softmax(y, dim=0)
    t[0] = 1

    epsilon = 1e-06
    loss = torch.sum(- t * torch.log(y + epsilon))
    print(y)
    print(t)
    print(loss)

def test_press_probability():
    x = torch.randn((2, 10, 5, 5), dtype=torch.float)
    y = torch.zeros((2, 5, 5), dtype=torch.long)
    mask = torch.zeros((2, 10, 5, 5), dtype=torch.float)

    kernel = 3

    p = torch.zeros((1, kernel, 1, 1), dtype=torch.long)
    p[0, :, 0, 0] = torch.arange(0, kernel)
    p = p.repeat(2, 1, 5, 5)

    y[0, 0, 0] = 5
    y[0, 1, 1] = 9

    mid = (y - kernel//2).unsqueeze(1)
    p = p + mid

    p[p <= 0] = 0
    p[p >= 10] = 9

    print(p[0, :, 0, 0])

    mask.scatter_(1, p, 1)
    x2 = x*mask
    print(x[0, :, 0, 0].view(-1))
    print(x2[0, :, 0, 0].view(-1))
    print(x2[0, :, 1, 1].view(-1))

def test_mask_cost_volume():
    x = torch.randn((2, 10, 5, 5), dtype=torch.float)
    y = torch.zeros((2, 5, 5), dtype=torch.long)

    y[0, 0, 0] = 5
    y[0, 1, 1] = 5
    mask = y == 5
    x = x.permute(1, 0, 2, 3)
    print(x[:, mask])

def test_confidnence():
    x = torch.randn((2, 10, 5, 5), dtype=torch.float)
    y = torch.zeros((2, 5, 5), dtype=torch.long)
    mask = torch.zeros((2, 10, 5, 5), dtype=torch.bool)

    y[0, 0, 0] = 1
    y[0, 0, 1] = 0
    y = y.unsqueeze(1)
    mask.scatter_(1, y, 1)

    x_mask = (x*mask).sum(dim=1)

    print(mask[0, :, 0, 0])
    print(x[0, :, 0, 0])
    print(x[0, :, 0, 1])
    print(x_mask)

test_confidnence()


