from GANet.basic import *
from GANet.function import *
import numpy as np

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()

    def forward(self, x, g):
        batch, channels, max_disparity, height, width = x.size()
        directions, weights = 4, 5

        g = g.view(batch, channels, directions, weights, height, width).contiguous()
        g = F.normalize(g, p=1, dim=3)
        x = SgaFunction.apply(x, g)  # output: cost_aggregation
        x = x.max(axis=2)[0]  # max in direction axis

        return x

class SGABlock(nn.Module):

    def __init__(self, channels):
        super(SGABlock, self).__init__()
        self.channels = channels

        self.sga = SGA()

        self.bn_relu = nn.Sequential(nn.BatchNorm3d(channels),
                                     nn.ReLU(inplace=True))
        self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)

        self.relu = nn.ReLU(inplace=True)

    # x: input cost
    # g: guidance
    def forward(self, x, g):
        rem = x
        x = self.sga(x, g)
        x = self.bn_relu(x)
        x = self.conv_refine(x)

        assert (x.size() == rem.size())
        x += rem
        x = self.relu(x)
        return x

class LGA(nn.Module):
    def __init__(self, kernel_size):
        super(LGA, self).__init__()
        self.kernel_size = kernel_size

    # x: input cost, no channel dimension (channel = 1, squeeze)
    # g: guidance
    def forward(self, x, g):
        g = F.normalize(g, p=1, dim=1)
        batch, max_disparity, height, width = x.size()
        g = g.view(batch, 3, self.kernel_size, self.kernel_size, height, width).contiguous()
        x = LgaFunction.apply(x, g)
        return x

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        with torch.cuda.device_of(x):
            disp = np.arange(self.maxdisp).reshape([1, self.maxdisp, 1, 1])
            disp = torch.tensor(disp, requires_grad=False).to(x.device).float()
            disp = disp.repeat(x.size(0), 1, x.size(2), x.size(3))
            x = torch.sum(x * disp, dim=1)
        return x

class DisparityClassRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityClassRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x, y):
        x = x.permute(1, 0, 2, 3)
        mask = (y < self.maxdisp - 1) & (y > 0)
        y = self.get_probability_volume(y[mask])
        loss = SoftmaxWithLoss.apply(x[:, mask], y)
        return loss

    def get_probability_volume(self, y):
        with torch.cuda.device_of(y):
            t = torch.zeros((self.maxdisp, y.size(0)), dtype=torch.float).to(y.device)
            index = y.long().unsqueeze(0)
            mid = y - index
            t.scatter_(0, index, 1 - mid)
            t.scatter_(0, index + 1, mid)
        return t

class SqueezeCost(nn.Module):
    def forward(self, cost, disp, kernel_size: int):
        assert cost.dtype == torch.float
        assert disp.dtype == torch.long

        with torch.cuda.device_of(cost):
            batch, max_disparity, height, width = cost.size()
            p = torch.zeros((1, kernel_size, 1, 1), dtype=torch.long).to(cost.device)
            mask = torch.zeros(cost.size(), dtype=torch.long).to(cost.device)

            p[0, :, 0, 0] = torch.arange(0, kernel_size)
            p = p.repeat(batch, 1, height, width)

            mid = (disp - kernel_size // 2).unsqueeze(1)
            p = p + mid

            p[p <= 0] = 0
            p[p >= max_disparity] = max_disparity - 1

            mask.scatter_(1, p, 1)
            cost_squeeze = cost * mask
            cost_squeeze = F.normalize(cost_squeeze, dim=1, p=1)
        return cost_squeeze

class SqueezeCostByGradient(nn.Module):
    def forward(self, cost, disp):
        assert cost.dtype == torch.float
        assert disp.dtype == torch.float

        with torch.cuda.device_of(cost):
            mask = torch.zeros(cost.size(), dtype=torch.uint8).to(cost.device)
            ganet_lib.cuda_cost_mask(cost, mask, disp)
            cost_squeeze = cost * mask
            cost_squeeze = F.normalize(cost_squeeze, dim=1, p=1)
        return mask, cost_squeeze

class CostVolume(nn.Module):
    def __init__(self, max_disparity):
        super(CostVolume, self).__init__()
        self.max_disparity = int(max_disparity)

    def forward(self, x, y):
        assert x.is_contiguous()
        with torch.cuda.device_of(x):
            # Size of batch, feature, disparity
            B, F, D = x.shape[0], x.shape[1], self.max_disparity

            # Size of height, width
            H, W = x.shape[2], x.shape[3]

            size = (B, F * 2, D, H, W)
            cost = torch.zeros([int(x) for x in size], requires_grad=False).to(x.device)

            for i in range(D):
                if i > 0:
                    cost[:, :F, i, :, i:] = x[:, :, :, i:]
                    cost[:, F:, i, :, i:] = y[:, :, :, :-i]
                else:
                    cost[:, :F, i, :, :] = x
                    cost[:, F:, i, :, :] = y

        return cost.contiguous()

class GDF4(nn.Module):
    def __init__(self, kernel_size):
        super(GDF4, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x, g):
        batch, channels, max_disparity, height, width = x.size()
        direction, weight_size = 4, self.kernel_size**2 + 1

        g = g.view(batch, channels, direction, weight_size, height, width).contiguous()
        g = F.normalize(g, p=1, dim=3)

        g0 = g[:, :, :, 0, :, :].contiguous()
        filter = g[:, :, :, 1:, :, :]
        filter = filter.view(batch, channels, direction, self.kernel_size, self.kernel_size, height, width).contiguous()

        x = GDF4_Function.apply(x, g0, filter)  # output: cost_aggregation
        x = x.view(batch, channels*4, max_disparity, height, width)

        return x

class GDF6(nn.Module):
    def __init__(self, kernel_size):
        super(GDF6, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x, g):
        batch, channels, max_disparity, height, width = x.size()
        direction, weight_size = 6, self.kernel_size**2 + 1

        g = g.view(batch, channels, direction, weight_size, height, width).contiguous()
        g = F.normalize(g, p=1, dim=3)

        g0 = g[:, :, :, 0, :, :].contiguous()
        filter = g[:, :, :, 1:, :, :]
        filter = filter.view(batch, channels, direction, self.kernel_size, self.kernel_size, height, width).contiguous()

        x = GDF6_Function.apply(x, g0, filter)  # output: cost_aggregation
        x = x.view(batch, channels*6, max_disparity, height, width)

        return x

class GDF4_Block(nn.Module):

    def __init__(self, channels, kernel_size):
        super(GDF4_Block, self).__init__()
        self.channels = channels

        self.sga = GDF4(kernel_size)
        self.conv = BasicConv(channels*4, channels, is_3d=True, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    # x: input cost
    # g: guidance
    def forward(self, x, g):
        rem = x
        x = self.sga(x, g)
        x = self.conv(x)

        assert (x.size() == rem.size())
        x = x + rem
        x = self.relu(x)
        return x

class GDF6_Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super(GDF6_Block, self).__init__()
        self.channels = channels

        self.gdf6 = GDF6(kernel_size)
        self.conv = BasicConv(channels*6, channels, is_3d=True, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    # x: input cost
    # g: guidance
    def forward(self, x, g):
        rem = x
        x = self.gdf6(x, g)
        x = self.conv(x)

        assert (x.size() == rem.size())
        x = x + rem
        x = self.relu(x)
        return x