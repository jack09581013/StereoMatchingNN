import torch
import torch.nn.functional as F
import utils.cost_volume as cv
import tools

batch = 2
height = 240
width = 576
max_disparity = 192
kernel_size = 3
P1 = 8 * 3 * kernel_size**2
P2 = 32 * 3 * kernel_size**2


left_image = torch.full((batch, 3, height, width), 1, dtype=torch.uint8).cuda()
right_image = torch.full((batch, 3, height, width), 2, dtype=torch.uint8).cuda()

cost = cv.cost_SAD(left_image, right_image, max_disparity, kernel_size)
disp = cv.sgm(cost, P1, P2)
print(disp[0, :, :])
