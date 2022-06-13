import torch
import torch.nn.functional as F
import utils.cost_volume as cv

batch = 1
height = 240
width = 576
disparity = (-32, 32)


left_image = torch.full((batch, 3, height, width), 1, dtype=torch.uint8).cuda()
right_image = torch.full((batch, 3, height, width), 2, dtype=torch.uint8).cuda()

cost = cv.calc_cost(left_image, right_image, disparity, 3, cv.CostMethod.CENSUS)
# print(cost.size())
print(cost[0, :, 0, 0])
