from CSPN.module import *
from GANet.basic import *
from GANet.GANet_small import Feature
from GANet.module import CostVolume, DisparityRegression
import torch.nn.functional as F

class CSPN(nn.Module):
    def __init__(self, max_disparity, cspn_kernel_size=3, cspn_round=12, fusion_kernel_size=3):
        super(CSPN, self).__init__()
        self.max_disparity = max_disparity
        self.cspn_kernel_size = cspn_kernel_size
        self.cspn_round = cspn_round
        self.fusion_kernel_size = fusion_kernel_size

        self.feature = Feature()
        self.cspf = CSPF(32)
        self.cost_volume = CostVolume(max_disparity//4)

        # Stacked Hourglass
        self.conv3d_0 = nn.Sequential(BasicConv(64, 32, kernel_size=3, padding=1, is_3d=True),
                                      BasicConv(32, 32, kernel_size=3, padding=1, is_3d=True))

        self.conv3d_1 = nn.Sequential(BasicConv(32, 32, kernel_size=3, padding=1, is_3d=True),
                                      BasicConv(32, 32, kernel_size=3, padding=1, is_3d=True))

        # Stack 1
        self.stack1_1 = nn.Sequential(BasicConv(32, 64, kernel_size=3, padding=1, stride=2, is_3d=True),
                                      BasicConv(64, 64, kernel_size=3, padding=1, is_3d=True))

        self.stack1_2 = nn.Sequential(BasicConv(64, 64, kernel_size=3, padding=1, stride=2, is_3d=True),
                                      BasicConv(64, 64, kernel_size=3, padding=1, is_3d=True))

        self.stack1_3 = BasicConv(64, 64, kernel_size=4, padding=1, stride=2, is_3d=True, deconv=True)
        self.stack1_4 = BasicConv(64, 32, kernel_size=4, padding=1, stride=2, is_3d=True, deconv=True)

        # Stack 2
        self.stack2_1 = nn.Sequential(BasicConv(32, 64, kernel_size=3, padding=1, stride=2, is_3d=True),
                                      BasicConv(64, 64, kernel_size=3, padding=1, is_3d=True))

        self.stack2_2 = nn.Sequential(BasicConv(64, 64, kernel_size=3, padding=1, stride=2, is_3d=True),
                                      BasicConv(64, 64, kernel_size=3, padding=1, is_3d=True))

        self.stack2_3 = BasicConv(64, 64, kernel_size=4, padding=1, stride=2, is_3d=True, deconv=True)
        self.stack2_4 = BasicConv(64, 32, kernel_size=4, padding=1, stride=2, is_3d=True, deconv=True)

        # Stack 3
        self.stack3_1 = nn.Sequential(BasicConv(32, 64, kernel_size=3, padding=1, stride=2, is_3d=True),
                                      BasicConv(64, 64, kernel_size=3, padding=1, is_3d=True))

        self.stack3_2 = nn.Sequential(BasicConv(64, 64, kernel_size=3, padding=1, stride=2, is_3d=True),
                                      BasicConv(64, 64, kernel_size=3, padding=1, is_3d=True))

        self.stack3_3 = BasicConv(64, 64, kernel_size=4, padding=1, stride=2, is_3d=True, deconv=True)
        self.stack3_4 = BasicConv(64, 32, kernel_size=4, padding=1, stride=2, is_3d=True, deconv=True)

        # 3D CSPN
        self.cspn3d_1 = CSPN_3D(32, self.cspn_kernel_size, self.cspn_round)
        self.cspn3d_2 = CSPN_3D(32, self.cspn_kernel_size, self.cspn_round)
        self.cspn3d_3 = CSPN_3D(32, self.cspn_kernel_size, self.cspn_round)

        self.disparity_fusion = DisparityFusion(self.max_disparity, self.fusion_kernel_size)


    def forward(self, x, y):
        D, H, W = self.max_disparity, x.size(2), x.size(3)

        x = self.feature(x)   # 32, H/4, W/4
        y = self.feature(y)   # 32, H/4, W/4

        x = self.cspf(x)  # 32, H/4, W/4
        y = self.cspf(y)  # 32, H/4, W/4

        x = self.cost_volume(x, y)  # 64, D/4, H/4, W/4

        out_conv3d_0 = self.conv3d_0(x)
        out_conv3d_1 = self.conv3d_1(out_conv3d_0) + out_conv3d_0

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

        out_cspn3d_1 = self.cspn3d_1(out_stack1_4)  # 1, D/4, H/4, W/4
        out_cspn3d_2 = self.cspn3d_2(out_stack2_4)  # 1, D/4, H/4, W/4
        out_cspn3d_3 = self.cspn3d_3(out_stack3_4)  # 1, D/4, H/4, W/4

        out_1 = F.interpolate(out_cspn3d_1, (D, H, W), mode='trilinear', align_corners=False).squeeze(1)
        out_2 = F.interpolate(out_cspn3d_2, (D, H, W), mode='trilinear', align_corners=False).squeeze(1)
        out_3 = F.interpolate(out_cspn3d_3, (D, H, W), mode='trilinear', align_corners=False).squeeze(1)

        fusion = self.disparity_fusion(out_1, out_2, out_3)  # out_1, out_2, out_3: D, H, W

        return fusion


class DisparityFusion(nn.Module):
    def __init__(self, max_disparity, kernel_size):
        super(DisparityFusion, self).__init__()
        self.kernel_size = kernel_size
        self.max_disparity = max_disparity

        self.conv_1 = BasicConv(1, kernel_size**2, kernel_size=3, padding=1, stride=1)
        self.conv_2 = BasicConv(1, kernel_size**2, kernel_size=3, padding=1, stride=1)
        self.conv_3 = BasicConv(1, kernel_size**2, kernel_size=3, padding=1, stride=1)

        self.disparity = DisparityRegression(max_disparity)

    def forward(self, out_1, out_2, out_3):
        batch, disparity, height, width = out_1.size()
        branch = 3

        # disparity regression
        out_1 = F.softmax(out_1, dim=1)
        out_2 = F.softmax(out_2, dim=1)
        out_3 = F.softmax(out_3, dim=1)

        out_1 = self.disparity(out_1).unsqueeze(1)  # batch, 1, height, width
        out_2 = self.disparity(out_2).unsqueeze(1)  # batch, 1, height, width
        out_3 = self.disparity(out_3).unsqueeze(1)  # batch, 1, height, width
        # disparity regression end

        # affinity matrix calculation
        branch_1 = self.conv_1(out_1).unsqueeze(1)  # batch, 1, kernel_size**2, height, width
        branch_2 = self.conv_2(out_2).unsqueeze(1)  # batch, 1, kernel_size**2, height, width
        branch_3 = self.conv_3(out_3).unsqueeze(1)  # batch, 1, kernel_size**2, height, width

        affinity_matrix = torch.cat([branch_1, branch_2, branch_3], dim=1)
        affinity_matrix = affinity_matrix.view(batch, branch * self.kernel_size ** 2, height, width)
        # affinity matrix calculation end

        # pyramid calculation
        out_1 = out_1.unsqueeze(2)  # batch, 1(channel), 1(branch), height, width
        out_2 = out_2.unsqueeze(2)  # batch, 1(channel), 1(branch), height, width
        out_3 = out_3.unsqueeze(2)  # batch, 1(channel), 1(branch), height, width

        pyramid = torch.cat([out_1, out_2, out_3], dim=2)  # concat in branch
        # pyramid calculation end

        fusion = cspn3d_fusion(affinity_matrix, pyramid, branch, self.kernel_size)

        return fusion.squeeze(1)  # batch, 1 (channel), height, width












