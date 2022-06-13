import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from CSPN.function import *


class CSPF(nn.Module):
    def __init__(self, channel):
        super(CSPF, self).__init__()
        self.guidance = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                      nn.BatchNorm2d(1),
                                      nn.ReLU(inplace=True))
        self.cspp_1 = CSPP(4)
        self.cspp_2 = CSPP(8)
        self.cspp_3 = CSPP(16)
        self.cspp_4 = CSPP(32)

        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channel),
                                    nn.ReLU(inplace=True))

        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channel),
                                    nn.ReLU(inplace=True))

        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channel),
                                    nn.ReLU(inplace=True))

        self.conv_4 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channel),
                                    nn.ReLU(inplace=True))
        self.cff = CFF(channel, 3)

    def forward(self, x):
        height, width = x.size()[2:4]
        g = self.guidance(x)

        branch_1 = self.cspp_1(x, g)
        branch_2 = self.cspp_2(x, g)
        branch_3 = self.cspp_3(x, g)
        branch_4 = self.cspp_4(x, g)

        branch_1 = self.conv_1(branch_1)
        branch_2 = self.conv_2(branch_2)
        branch_3 = self.conv_3(branch_3)
        branch_4 = self.conv_4(branch_4)

        branch_1 = F.interpolate(branch_1, (height, width), mode='bilinear', align_corners=False).unsqueeze(2)
        branch_2 = F.interpolate(branch_2, (height, width), mode='bilinear', align_corners=False).unsqueeze(2)
        branch_3 = F.interpolate(branch_3, (height, width), mode='bilinear', align_corners=False).unsqueeze(2)
        branch_4 = F.interpolate(branch_4, (height, width), mode='bilinear', align_corners=False).unsqueeze(2)

        pyramid = torch.cat([branch_1, branch_2, branch_3, branch_4], dim=2)
        fusion = self.cff(x, pyramid)

        return fusion


class CSPP(nn.Module):
    def __init__(self, kernel_size, epsilon=1e-06):
        super(CSPP, self).__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def forward(self, x, g):
        batch, channel, height, width = x.size()
        pooling_height = height // self.kernel_size
        pooling_width = width // self.kernel_size

        g = F.unfold(g, kernel_size=self.kernel_size, stride=self.kernel_size)
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.kernel_size)

        # block normalize
        g = g.abs()
        g = g / (g.sum(dim=1).unsqueeze(1) + self.epsilon)

        g = g.repeat(1, channel, 1)
        y = g * x
        y = y.view(batch, channel, self.kernel_size ** 2, -1)
        y = y.sum(dim=2)  # output dimension: batch, channel, pooling height * pooling width
        y = y.view(batch, channel, pooling_height, pooling_width)
        return y


class CFF(nn.Module):
    def __init__(self, channel, kernel_size):
        super(CFF, self).__init__()
        self.kernel_size = kernel_size

        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel, kernel_size ** 2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(kernel_size ** 2),
            nn.ReLU(inplace=True))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel, kernel_size ** 2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(kernel_size ** 2),
            nn.ReLU(inplace=True))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel, kernel_size ** 2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(kernel_size ** 2),
            nn.ReLU(inplace=True))

        self.conv_4 = nn.Sequential(
            nn.Conv2d(channel, kernel_size ** 2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(kernel_size ** 2),
            nn.ReLU(inplace=True))

    def forward(self, x, pyramid):
        batch, channel, height, width = x.size()
        branch = 4

        branch_1 = self.conv_1(x).unsqueeze(1)
        branch_2 = self.conv_2(x).unsqueeze(1)
        branch_3 = self.conv_3(x).unsqueeze(1)
        branch_4 = self.conv_4(x).unsqueeze(1)

        affinity_matrix = torch.cat([branch_1, branch_2, branch_3, branch_4], dim=1)
        affinity_matrix = affinity_matrix.view(batch, branch * self.kernel_size ** 2, height, width)

        fusion = cspn3d_fusion(affinity_matrix, pyramid, branch, self.kernel_size)
        return fusion


class CSPN_3D(nn.Module):
    def __init__(self, channel, kernel_size, round):
        super(CSPN_3D, self).__init__()
        self.kernel_size = kernel_size
        self.round = round
        self.guidance = nn.Conv3d(channel, kernel_size ** 3 - 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv = nn.Conv3d(channel, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        g = self.guidance(x)
        x = self.conv(x)
        x = cspn3d(x, g, self.kernel_size, self.round)
        return x
