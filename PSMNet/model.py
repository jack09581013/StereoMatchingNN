import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.input_adjust = None
        if stride != 1 or in_planes != planes:
            self.input_adjust = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                              nn.BatchNorm2d(planes))

        padding = dilation if dilation > 1 else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, padding=padding, stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, 3, padding=padding, dilation=dilation, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.input_adjust is not None:
            x = self.input_adjust(x)
        out = out + x
        return out


class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     nn.Conv2d(128, 32, 1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     nn.Conv2d(128, 32, 1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     nn.Conv2d(128, 32, 1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     nn.Conv2d(128, 32, 1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        # planes 320 = 32*4(branches) + 128(conv4_3) + 64(conv2_16)
        self.fusion = nn.Sequential(nn.Conv2d(320, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 32, 1, bias=False))

    def forward(self, conv2_16, conv4_3):
        H, W = conv4_3.size()[2:4]

        # output for 4 branches
        out_branch1 = Func.interpolate(self.branch1(conv4_3), (H, W), mode='bilinear', align_corners=True)
        out_branch2 = Func.interpolate(self.branch2(conv4_3), (H, W), mode='bilinear', align_corners=True)
        out_branch3 = Func.interpolate(self.branch3(conv4_3), (H, W), mode='bilinear', align_corners=True)
        out_branch4 = Func.interpolate(self.branch4(conv4_3), (H, W), mode='bilinear', align_corners=True)

        concat = torch.cat((conv2_16, conv4_3, out_branch1, out_branch2, out_branch3, out_branch4), 1)
        final_out = self.fusion(concat)

        return final_out


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.conv0_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.conv0_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.conv0_3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.conv1_x = conv1_x()
        self.conv2_x = conv2_x()
        self.conv3_x = conv3_x()
        self.conv4_x = conv4_x()

        self.SPP = SpatialPyramidPooling()

    def forward(self, x):
        out = self.conv0_1(x)
        out = self.conv0_2(out)
        out = self.conv0_3(out)
        out = self.conv1_x(out)
        out_conv2_16 = self.conv2_x(out)
        out = self.conv3_x(out_conv2_16)
        out_conv4_3 = self.conv4_x(out)
        out = self.SPP(out_conv2_16, out_conv4_3)

        return out


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.disp = torch.tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]), requires_grad=False).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


def conv1_x():
    layers = []
    for i in range(3):
        layers.append(ResidualBlock(32, 32))
    return nn.Sequential(*layers)


def conv2_x():
    layers = []
    layers.append(ResidualBlock(32, 64, stride=2))
    for i in range(15):
        layers.append(ResidualBlock(64, 64))
    return nn.Sequential(*layers)


def conv3_x():
    layers = []
    layers.append(ResidualBlock(64, 128, dilation=2))
    for i in range(2):
        layers.append(ResidualBlock(128, 128, dilation=2))
    return nn.Sequential(*layers)


def conv4_x():
    layers = []
    for i in range(3):
        layers.append(ResidualBlock(128, 128, dilation=4))
    return nn.Sequential(*layers)


def conv3d(in_planes, planes, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Conv3d(in_planes, planes, kernel_size, padding=padding, stride=stride, bias=False),
                         nn.BatchNorm3d(planes),
                         nn.ReLU(inplace=True))


def deconv3d(in_planes, planes, kernel_size=4, stride=2):
    return nn.Sequential(nn.ConvTranspose3d(in_planes, planes, kernel_size, padding=1, stride=stride, bias=False),
                         nn.BatchNorm3d(planes))


class PSMNet(nn.Module):
    def __init__(self, max_disparity):
        super(PSMNet, self).__init__()
        self.max_disparity = max_disparity
        self.feature_extraction = FeatureExtraction()

        # Stacked Hourglass
        self.conv3d_0 = nn.Sequential(conv3d(64, 32),
                                      conv3d(32, 32))

        self.conv3d_1 = nn.Sequential(conv3d(32, 32),
                                      conv3d(32, 32))

        # Stack 1
        self.stack1_1 = nn.Sequential(conv3d(32, 64, stride=2),
                                      conv3d(64, 64))

        self.stack1_2 = nn.Sequential(conv3d(64, 64, stride=2),
                                      conv3d(64, 64))

        self.stack1_3 = deconv3d(64, 64)
        self.stack1_4 = deconv3d(64, 32)

        # Stack 2
        self.stack2_1 = nn.Sequential(conv3d(32, 64, stride=2),
                                      conv3d(64, 64))

        self.stack2_2 = nn.Sequential(conv3d(64, 64, stride=2),
                                      conv3d(64, 64))

        self.stack2_3 = deconv3d(64, 64)
        self.stack2_4 = deconv3d(64, 32)

        # Stack 3
        self.stack3_1 = nn.Sequential(conv3d(32, 64, stride=2),
                                      conv3d(64, 64))

        self.stack3_2 = nn.Sequential(conv3d(64, 64, stride=2),
                                      conv3d(64, 64))

        self.stack3_3 = deconv3d(64, 64)
        self.stack3_4 = deconv3d(64, 32)

        # Output
        self.output_1 = nn.Sequential(conv3d(32, 32),
                                      conv3d(32, 1))

        self.output_2 = nn.Sequential(conv3d(32, 32),
                                      conv3d(32, 1))

        self.output_3 = nn.Sequential(conv3d(32, 32),
                                      conv3d(32, 1))

    def forward(self, left_image, right_image):
        out_l = self.feature_extraction(left_image)
        out_r = self.feature_extraction(right_image)

        # Size of batch, feature, disparity
        B, F, D = left_image.shape[0], out_l.size()[1], self.max_disparity

        # Size of height, width
        H, W = left_image.shape[2], left_image.shape[3]

        size = (B, F * 2, D / 4, H / 4, W / 4)
        cost = torch.zeros([int(x) for x in size], requires_grad=False).cuda()

        for i in range(int(D / 4)):
            if i > 0:
                cost[:, :F, i, :, i:] = out_l[:, :, :, i:]
                cost[:, F:, i, :, i:] = out_r[:, :, :, :-i]
            else:
                cost[:, :F, i, :, :] = out_l
                cost[:, F:, i, :, :] = out_r

        out_conv3d_0 = self.conv3d_0(cost)
        out_conv3d_1 = self.conv3d_1(out_conv3d_0)

        # Stack 1
        out_stack1_1 = self.stack1_1(out_conv3d_1)
        out_stack1_2 = self.stack1_2(out_stack1_1)
        out_stack1_3 = self.stack1_3(out_stack1_2) + out_stack1_1  # Green line
        out_stack1_4 = self.stack1_4(out_stack1_3) + out_conv3d_1  # Orange line

        # Stack 2
        out_stack2_1 = self.stack2_1(out_stack1_4) + out_stack1_3  # Red line
        out_stack2_2 = self.stack2_2(out_stack2_1)
        out_stack2_3 = self.stack2_3(out_stack2_2) + out_stack1_1  # Green line
        out_stack2_4 = self.stack2_4(out_stack2_3) + out_conv3d_1  # Orange line

        # Stack 3
        out_stack3_1 = self.stack3_1(out_stack2_4) + out_stack2_3  # Red line
        out_stack3_2 = self.stack3_2(out_stack3_1)
        out_stack3_3 = self.stack3_3(out_stack3_2) + out_stack1_1  # Green line
        out_stack3_4 = self.stack3_4(out_stack3_3) + out_conv3d_1  # Orange line

        out_1 = self.output_1(out_stack1_4)
        out_2 = self.output_2(out_stack2_4) + out_1
        out_3 = self.output_3(out_stack3_4) + out_2

        if self.training:
            out_1 = -out_1
            out_2 = -out_2
            out_3 = -out_3

            out_1 = Func.interpolate(out_1, (D, H, W), mode='trilinear', align_corners=True)
            out_2 = Func.interpolate(out_2, (D, H, W), mode='trilinear', align_corners=True)
            out_3 = Func.interpolate(out_3, (D, H, W), mode='trilinear', align_corners=True)

            out_1 = torch.squeeze(out_1, 1)
            out_2 = torch.squeeze(out_2, 1)
            out_3 = torch.squeeze(out_3, 1)

            out_1 = Func.softmax(out_1, dim=1)
            out_2 = Func.softmax(out_2, dim=1)
            out_3 = Func.softmax(out_3, dim=1)

            out_1 = DisparityRegression(self.max_disparity)(out_1)
            out_2 = DisparityRegression(self.max_disparity)(out_2)
            out_3 = DisparityRegression(self.max_disparity)(out_3)

            return out_1, out_2, out_3

        else:
            out_3 = -out_3
            out_3 = Func.interpolate(out_3, (D, H, W), mode='trilinear', align_corners=True)
            out_3 = torch.squeeze(out_3, 1)
            out_3 = Func.softmax(out_3, dim=1)
            out_3 = DisparityRegression(self.max_disparity)(out_3)
            return out_3
