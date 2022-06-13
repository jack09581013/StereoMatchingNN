from GANet.module import *
from GANet.function import *
import GDFNet.GDFNet_dc6
import utils

class GDFNet_dc6f(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDFNet_dc6f, self).__init__()
        self.max_disparity = max_disparity
        self.model = GDFNet.GDFNet_dc6.GDFNet_dc6(max_disparity)
        self.flip = False

    def forward(self, x, y):
        self.model.flip = self.flip

        if self.training:
            if self.flip:
                cost1, cost2, cost3 = self.model(x, y)
                cost1f = FlipCost.apply(cost1)[..., self.max_disparity:]
                cost2f = FlipCost.apply(cost2)[..., self.max_disparity:]
                cost3f = FlipCost.apply(cost3)[..., self.max_disparity:]
                return cost1f, cost2f, cost3f

            else:
                cost1, cost2, cost3 = self.model(x, y)
                return cost1, cost2, cost3
        else:
            if self.flip:
                cost = self.model(x, y)
                # cost = FlipCost.apply(cost)
                return cost

            else:
                cost = self.model(x, y)
                return cost