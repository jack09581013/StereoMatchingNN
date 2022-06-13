import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
import os

def cspn2d(H0, K, kernel_size, round, debug=False, epsilon=1e-06):
    assert H0.dim() == 4

    if debug:
        H_last = H0

    batch, channels, height, width = H0.size()
    H = padding2d(H0, kernel_size)
    K = K.view(batch, kernel_size ** 2 - 1, channels, height, width)
    K = K / (K.abs().sum(dim=1).unsqueeze(1) + epsilon)
    k0 = (1.0 - K.sum(dim=1))
    K = K.view(batch, (kernel_size ** 2 - 1)*channels, height, width)

    for r in range(round):
        H1 = k0 * H0 + (H * K).sum(dim=1).unsqueeze(1)
        if debug:
            diff = (H1 - H_last).abs().view(-1).mean()
            H_last = H1
            print('{:e}'.format(diff))
            # print(H)
        if r < round - 1:
            H = padding2d(H1, kernel_size)
        else:
            H = H1
    return H

def cspn3d(H0, K, kernel_size, round, debug=False, epsilon=1e-06):
    assert H0.dim() == 5

    if debug:
        H_last = H0

    batch, channels, disparity, height, width = H0.size()
    H = padding3d(H0, kernel_size)
    K = K.view(batch, kernel_size**3 - 1, channels, disparity, height, width)

    K = K / (K.abs().sum(dim=1).unsqueeze(1) + epsilon)
    k0 = (1.0 - K.sum(dim=1))

    K = K.view(batch, (kernel_size**3 - 1)*channels, disparity, height, width)

    for r in range(round):
        H1 = k0 * H0 + (H * K).sum(dim=1).unsqueeze(1)
        if debug:
            diff = (H1 - H_last).abs().view(-1).mean()
            H_last = H1
            print('{:e}'.format(diff))
            # print(H1[0])
        if r < round - 1:
            H = padding3d(H1, kernel_size)
        else:
            H = H1
    return H

def cspn3d_fusion(K, H, branch, kernel_size, epsilon=1e-06):
    assert H.dim() == 5
    assert K.size(1) == branch * kernel_size**2
    # K size: (batch, branch * kernel_size**2, height, width)

    H = padding3d_fusion(H, kernel_size)
    K = K.abs()
    K = K / (K.sum(dim=1).unsqueeze(1) + epsilon)

    K = K.unsqueeze(1)
    H = (H * K).sum(dim=2)

    return H

def padding2d(x, kernel_size):
    assert x.dim() == 4

    padding = get_kernel_padding_2d(kernel_size)
    feature_map = []
    for i in range(len(padding)):
        padding_map = F.pad(x, padding[i])
        feature_map.append(padding_map)
    feature_map = torch.cat(feature_map, 1)  # concat in channel
    return feature_map

def padding3d(x, kernel_size):
    assert x.dim() == 5

    padding = get_kernel_padding_3d(kernel_size)
    feature_map = []
    for i in range(len(padding)):
        padding_map = F.pad(x, padding[i])
        feature_map.append(padding_map)
    feature_map = torch.cat(feature_map, 1)  # concat in channel
    return feature_map

def padding3d_fusion(x, kernel_size):
    assert x.dim() == 5
    batch, channel, branch, height, width = x.size()

    padding = get_kernel_padding_3d_fusion(branch, kernel_size)
    feature_map = []
    for i in range(len(padding)):
        padding_map = F.pad(x, padding[i]).unsqueeze(2)
        feature_map.append(padding_map)
    feature_map = torch.cat(feature_map, 2)  # concat in branch
    return feature_map[:, :, :, branch//2, :, :]

def get_kernel_padding_2d(kernel_size):
    assert kernel_size % 2 == 1, 'kernel size must be odd a number'
    padding = []
    mid = kernel_size//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == mid and j == mid:
                continue
            row_padding = (mid - i, i - mid)
            column_padding = (mid - j, j - mid)
            padding.append([*column_padding, *row_padding])
    return padding

def get_kernel_padding_3d(kernel_size):
    assert kernel_size % 2 == 1, 'kernel size must be odd a number'
    padding = []
    mid = kernel_size//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                if i == mid and j == mid and k == mid:
                    continue
                disparity_padding = (mid - i, i - mid)
                row_padding = (mid - j, j - mid)
                column_padding = (mid - k, k - mid)
                padding.append([*column_padding, *row_padding, *disparity_padding])
    return padding

def get_kernel_padding_3d_fusion(branch, kernel_size):
    assert kernel_size % 2 == 1, 'kernel size must be odd a number'
    padding = []
    mid = kernel_size//2
    branch_mid = branch//2
    for i in range(branch):
        for j in range(kernel_size):
            for k in range(kernel_size):
                branch_padding = (branch_mid - i, i - branch_mid)
                row_padding = (mid - j, j - mid)
                column_padding = (mid - k, k - mid)
                padding.append([*column_padding, *row_padding, *branch_padding])
    return padding