from GANet.module import *

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2))

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)

        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)

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

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)

        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv13 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv14 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_gdf1 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gdf2 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gdf3 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_gdf11 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gdf12 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gdf13 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gdf14 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

    def forward(self, x):
        x = self.conv0(x)  # H, W
        rem = x

        # gdf1, gdf2, gdf3: 1920, H/4, W/4
        # 1920 = 32*6*10
        x = self.conv1(x)
        gdf1 = self.weight_gdf1(x)
        x = self.conv2(x)
        gdf2 = self.weight_gdf2(x)
        x = self.conv3(x)
        gdf3 = self.weight_gdf3(x)

        # gdf11, gdf12, gdf13, gdf14: 2880, H/8, W/8
        # 2880 = 48*6*10
        x = self.conv11(x)
        gdf11 = self.weight_gdf11(x)
        x = self.conv12(x)
        gdf12 = self.weight_gdf12(x)
        x = self.conv13(x)
        gdf13 = self.weight_gdf13(x)
        x = self.conv14(x)
        gdf14 = self.weight_gdf14(x)

        # lg1: 75, H, W
        # 75 = 3*5*5
        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)

        return dict([
            ('gdf1', gdf1),
            ('gdf2', gdf2),
            ('gdf3', gdf3),
            ('gdf11', gdf11),
            ('gdf12', gdf12),
            ('gdf13', gdf13),
            ('gdf14', gdf14),
            ('lg1', lg1),
            ('lg2', lg2)])

class Disparity(nn.Module):

    def __init__(self, max_disparity=192):
        super(Disparity, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.disparity = DisparityRegression(self.max_disparity)

    def forward(self, x):
        x = F.interpolate(self.conv32x1(x), scale_factor=4, mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        x = F.softmax(x, dim=1)  # D, H, W
        return self.disparity(x)

class DisparityAggregation(nn.Module):

    def __init__(self, max_disparity=192):
        super(DisparityAggregation, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.lga = LGA(5)
        self.disparity = DisparityRegression(self.max_disparity)

    def forward(self, x, lg1, lg2):
        x = F.interpolate(self.conv32x1(x), scale_factor=4, mode='trilinear', align_corners=False)
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

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)

        self.conv1b = Conv2x(32, 48, is_3d=True)
        self.conv2b = Conv2x(48, 64, is_3d=True)
        self.deconv2b = Conv2x(64, 48, deconv=True, is_3d=True)
        self.deconv1b = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)

        self.gdf1 = GDF6_Block(32, 3)
        self.gdf2 = GDF6_Block(32, 3)
        self.gdf3 = GDF6_Block(32, 3)
        self.gdf11 = GDF6_Block(48, 3)
        self.gdf12 = GDF6_Block(48, 3)
        self.gdf13 = GDF6_Block(48, 3)
        self.gdf14 = GDF6_Block(48, 3)

        self.disp0 = Disparity(self.maxdisp)
        self.disp1 = Disparity(self.maxdisp)
        self.disp2 = DisparityAggregation(self.maxdisp)

    def forward(self, x, g):
        x = self.conv_start(x)

        # Part 1
        # x: 32, D/4, H/4, W/4
        # gdf1: 1920, H/4, W/4
        x = self.gdf1(x, g['gdf1'])
        rem0 = x

        if self.training:
            disp0 = self.disp0(x)

        x = self.conv1a(x)
        x = self.gdf11(x, g['gdf11'])
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.deconv2a(x, rem1)
        x = self.gdf12(x, g['gdf12'])
        rem1 = x
        x = self.deconv1a(x, rem0)

        # Part 2
        x = self.gdf2(x, g['gdf2'])
        rem0 = x
        if self.training:
            disp1 = self.disp1(x)

        x = self.conv1b(x, rem1)
        x = self.gdf13(x, g['gdf13'])
        rem1 = x
        x = self.conv2b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.gdf14(x, g['gdf14'])
        x = self.deconv1b(x, rem0)

        # Part 3
        x = self.gdf3(x, g['gdf3'])

        # x: 32, D/4, H/4, W/4
        # lg1: 75, H, W
        # lg2: 75, H, W
        disp2 = self.disp2(x, g['lg1'], g['lg2'])
        if self.training:
            return disp0, disp1, disp2
        else:
            return disp2

class GDFNet_md6(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDFNet_md6, self).__init__()
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
        self.cost_volume = CostVolume(max_disparity/4)
        self.cost_aggregation = CostAggregation(self.max_disparity)
        self.flip = False

    def forward(self, x, y):
        if self.flip:
            flip_x = x.data.cpu().numpy()
            flip_y = y.data.cpu().numpy()
            flip_x = torch.tensor(flip_x[..., ::-1].copy()).cuda()
            flip_y = torch.tensor(flip_y[..., ::-1].copy()).cuda()
            x = flip_y
            y = flip_x

        g = self.conv_start(x)  # 32, H, W
        x = self.feature(x)
        y = self.feature(y)

        rem = x
        x = self.conv_x(x)  # 32, H/4, W/4
        y = self.conv_y(y)  # 32, H/4, W/4

        x = self.cost_volume(x, y)  # 64, D/4, H/4, W/4

        x1 = self.conv_refine(rem)  # 32, H/4, W/4

        # 32, H, W
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)

        g = torch.cat((g, x1), 1)  # 64, H, W

        # gdf1, gdf2, gdf3: 1920, H/4, W/4
        # 1920 = 32*6*10
        # gdf11, gdf12, gdf13, gdf14: 2880, H/8, W/8
        # 2880 = 48*6*10
        # lg1: 75, H, W
        # lg2: 75, H, W
        g = self.guidance(g)

        return self.cost_aggregation(x, g)