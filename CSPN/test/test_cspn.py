import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from CSPN.function import *
from CSPN.module import *
from CSPN.cspn import CSPN
from GANet.GANet_small import GANetSmall
import os

# print('x {:.3f}'.format(x.view(-1).mean()))

def normalize(k8):
    sum_k = k8.abs().sum(dim=1)
    k8 /= sum_k
    return k8

def cspn_v1(K, H, round=10):
    channels = K.size(0)
    height, width = K.size()[3:5]
    H = F.pad(H, [1, 1, 1, 1], value=0)
    H0 = H
    Ht = H.clone()
    k8 = K.view(channels, 9, height, width)[:, [0, 1, 2, 3, 5, 6, 7, 8], :, :]

    k8[...] = normalize(k8)
    sum_k = k8.sum(dim=1)
    k0 = 1.0 - sum_k

    K.view(channels, 9, height, width)[:, [0, 1, 2, 3, 5, 6, 7, 8], :, :] = k8[...]
    K.view(channels, 9, height, width)[:, 4, :, :] = 0

    for r in range(round):
        Ht1 = torch.zeros(H.size())
        for i in range(height):
            for j in range(width):
                conv = (Ht[:, i:i + 3, j:j + 3] * K[:, :, :, i, j]).view(-1).sum()
                Ht1[:, i + 1, j + 1] = conv + k0[:, i, j] * H0[:, i + 1, j + 1]
        diff = (Ht1 - Ht).abs().view(-1).sum()
        Ht = Ht1
        # print(Ht)
        print('{:e}'.format(diff))
    return Ht[:, 1:-1, 1:-1]

def test_cspn_v1():
    height = 5
    width = 5
    channels = 1

    H = torch.randn((channels, height, width))
    K = torch.randn((channels, 3, 3, height, width))

    print(H)
    Ht = cspn_v1(K, H)
    print(Ht)

def test_cspn2d():
    batch = 2
    channels = 2
    height = 5
    width = 5
    kernel_size = 3

    H = torch.randn((batch, channels, height, width), requires_grad=True, device='cuda')
    K = torch.randn((batch, channels*(kernel_size**2 - 1), height, width), requires_grad=True, device='cuda')

    print(H[0])
    H = cspn2d(H, K, kernel_size, 20, debug=True)
    print(H[0])

def test_cspn3d():
    batch = 1
    channels = 2
    disparity = 192//4
    height = 256//4
    width = 512//4
    # disparity = 1
    # height = 5
    # width = 5
    kernel_size = 3

    H = torch.randn((batch, channels, disparity, height, width), requires_grad=True, device='cuda')
    K = torch.randn((batch, channels*(kernel_size**3 - 1), disparity, height, width), requires_grad=True, device='cuda')

    # print(H[0])
    H = cspn3d(H, K, kernel_size, 20, debug=True)
    # print(H[0])

def test_cspn3d_fusion():
    batch = 2
    channels = 8
    branch = 4
    height = 5
    width = 5
    kernel_size = 3

    H = torch.randn((batch, channels, branch, height, width))
    K = torch.randn((batch, branch, kernel_size**2, height, width))
    K = K.view(batch, branch * kernel_size**2, height, width)

    h = H[0, :, :, 0:3, 0:3].reshape(channels, -1)  # channel, branch*kernel*kernel = 4*3*3 = 36
    k = K[0, :, 1, 1]   # branch*kernel*kernel = 4*3*3 = 36

    k = k.abs()
    k = k / (k.sum(dim=0) + 1e-06)  # sum in kernel dim
    k = k.unsqueeze(0)  # 1, branch*kernel*kernel
    conv = (k * h).sum(dim=1)
    print(conv)

    H = cspn3d_fusion(K, H, branch, kernel_size)
    print(H[0, :, 1, 1])

def test_cspp():
    batch = 2
    channels = 8
    height = 10
    width = 10
    kernel_size = 3

    x = torch.randn((batch, channels, height, width), requires_grad=True)
    g = torch.randn((batch, 1, height, width), requires_grad=True)

    g1 = g[0, :, 3:6, 3:6].abs()
    g1 = g1 / g1.view(-1).sum(dim=0)

    conv = x[0, :, 3:6, 3:6] * g1
    sum = conv.view(-1, kernel_size**2).sum(dim=1)
    print(sum)

    spp = CSPP(kernel_size)
    y = spp(x, g)
    print(y[0, :, 1, 1])

def test_padding2d():
    batch = 2
    channels = 1
    height = 5
    width = 5
    kernel_size = 3

    H = torch.randn((batch, channels, height, width))

    print(H[0])

    H = padding2d(H, kernel_size)
    print(H[0, :, 1, 1])

def test_padding3d():
    batch = 2
    channels = 1
    disparity = 5
    height = 5
    width = 5
    kernel_size = 3

    H = torch.randn((batch, channels, disparity, height, width))

    print(H[0, 0, 0:3, 0:3, 1:4])

    H = padding3d(H, kernel_size)
    print(H[0, :, 1, 1, 2])

def test_padding3d_fusion():
    batch = 2
    channels = 2
    branch = 4
    height = 5
    width = 5
    kernel_size = 3

    H = torch.randn((batch, channels, branch, height, width))

    print(H[0, 0, :, -4:-1, -4:-1])

    H = padding3d_fusion(H, kernel_size)
    print(H.size())
    print(H[0, 0, :, -3, -3])

def test_csff():
    batch = 2
    channels = 32
    height = 256//4
    width = 512//4

    x = torch.randn((batch, channels, height, width), device='cuda')
    y = torch.randn((batch, channels, height, width), device='cuda')

    model = CSPF(channels).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    for i in range(100):
        optimizer.zero_grad()
        p = model(x)
        loss = F.mse_loss(p, y)
        loss.backward()
        # os.system('nvidia-smi')  # 1973MiB
        optimizer.step()
        print('loss = {:.3f}'.format(loss))
        if torch.isnan(loss):
            break

def test_cspn3d_module():
    batch = 2
    channels = 32
    disparity = 192 // 4
    height = 256 // 4
    width = 512 // 4
    kernel_size = 3
    round = 20

    x = torch.randn((batch, channels, disparity, height, width), requires_grad=True, device='cuda')
    y = torch.randn((batch, 1, disparity, height, width), requires_grad=True, device='cuda')

    model = CSPN_3D(channels, kernel_size, round).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    for i in range(1):
        optimizer.zero_grad()
        p = model(x)
        loss = F.mse_loss(p, y)
        loss.backward()
        os.system('nvidia-smi')  # 3223MiB
        optimizer.step()
        print('loss = {:.3f}'.format(loss))
        if torch.isnan(loss):
            break

def test_cspn():
    batch = 1
    channels = 3
    disparity = 192//2
    height = 256//2
    width = 512//2

    x1 = torch.randn((batch, channels, height, width), device='cuda')
    x2 = torch.randn((batch, channels, height, width), device='cuda')
    y = torch.randint(0, disparity, (batch, height, width), device='cuda', dtype=torch.float)
    print('y min={:.3f}, max={:.3f}'.format(y.view(-1).min(), y.view(-1).max()))

    model = CSPN(disparity, 3, 12, 3).cuda()
    # model = GANetSmall(disparity).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    for i in range(100):
        optimizer.zero_grad()
        p = model(x1, x2)
        print('p min={:.3f}, max={:.3f}'.format(p.view(-1).min(), p.view(-1).max()))
        loss = F.mse_loss(p, y)
        loss.backward()
        # os.system('nvidia-smi')  # 5843MiB
        optimizer.step()
        print('loss = {:.3f}'.format(loss))
        if torch.isnan(loss):
            break


# test_cspn2d()
# test_padding2d()

test_cspn3d()
# test_padding3d()

# test_cspn3d_fusion()
# test_padding3d_fusion()

# test_cspp()
# test_csff()
# test_cspn3d_module()

# test_cspn()