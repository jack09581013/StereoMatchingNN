from GANet.module import *

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.weight_sg1 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_sg2 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

    def forward(self, x):
        x = self.conv0(x)
        rem = x
        x = self.conv1(x)

        # sg1: 640, H/3, W/3
        # 640 = 32*4*5
        sg1 = self.weight_sg1(x)

        x = self.conv2(x)

        # sg1: 640, H/3, W/3
        sg2 = self.weight_sg2(x)

        # lg1: 75, H, W
        # 75 = 3*5*5
        lg1 = self.weight_lg1(rem)

        # lg2: 75, H, W
        lg2 = self.weight_lg2(rem)

        return {
            'sg1': sg1,
            'sg2': sg2,
            'lg1': lg1,
            'lg2': lg2
        }

class DisparityAggregation(nn.Module):

    def __init__(self, max_disparity=192):
        super(DisparityAggregation, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.lga = LGA(5)
        self.disparity = DisparityRegression(self.max_disparity)

    def forward(self, x, lg1, lg2):
        x = F.interpolate(self.conv32x1(x), [self.max_disparity, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        x = self.lga(x, lg1)     # D, H, W
        x = F.softmax(x, dim=1)  # D, H, W
        x = self.lga(x, lg2)     # D, H, W
        x = F.normalize(x, p=1, dim=1)  # D, H, W
        return self.disparity(x)

class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)
        self.conv0 = BasicConv(32, 32, is_3d=True, kernel_size=3, padding=1)

        self.sga1 = SGABlock(32)
        self.sga2 = SGABlock(32)
        self.disp = DisparityAggregation(self.maxdisp)

    def forward(self, x, g):
        x = self.conv_start(x)

        # x: 32, D/3, H/3, W/3
        # sg1: 640, H/3, W/3
        x = self.sga1(x, g['sg1'])

        x = self.conv0(x)

        # x: 32, D/3, H/3, W/3
        # sg1: 640, H/3, W/3
        x = self.sga2(x, g['sg2'])

        # x:
        # lg1: 75, H, W
        # lg2: 75, H, W
        x = self.disp(x, g['lg1'], g['lg2'])

        return x

class GANet(nn.Module):
    def __init__(self, max_disparity=192):
        super(GANet, self).__init__()
        self.max_disparity = max_disparity

        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.guidance = Guidance()
        self.feature = Feature()
        self.cost_volume = CostVolume(max_disparity/3)
        self.cost_aggregation = CostAggregation(self.max_disparity)

    def forward(self, x, y):
        g = self.conv_start(x)  # 32, H, W
        x = self.feature(x)
        y = self.feature(y)

        rem = x
        x = self.conv_x(x)  # 32, H/3, W/3
        y = self.conv_y(y)  # 32, H/3, W/3

        x = self.cost_volume(x, y)  # 64, D/3, H/3, W/3

        x1 = self.conv_refine(rem)  # 32, H/3, W/3

        # 32, H, W
        x1 = F.interpolate(x1, [x1.size()[2] * 3, x1.size()[3] * 3], mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)

        g = torch.cat((g, x1), 1)  # 64, H, W

        # sg1: 640, H/3, W/3
        # sg2: 640, H/3, W/3
        # lg1: 75, H, W
        # lg2: 75, H, W
        g = self.guidance(g)

        return self.cost_aggregation(x, g)