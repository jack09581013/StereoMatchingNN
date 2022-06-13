import torch
import torch.nn.functional as F
import ganet_lib
import tools
import utils

def calc_cost(left_image, right_image, disparity, kenel_size, method):
    batch, channel, height, width = left_image.size()
    min_disparity, max_disparity = disparity
    assert left_image.is_contiguous()
    assert right_image.is_contiguous()
    assert max_disparity > min_disparity
    assert kenel_size % 2 == 1, 'kenel_size must be odd number'

    with torch.cuda.device_of(left_image):
        disparity_range = max_disparity - min_disparity
        cost = torch.zeros((batch, disparity_range, height, width), dtype=torch.float).to(left_image.device).contiguous()
        ganet_lib.cuda_calc_cost(left_image, right_image, cost, kenel_size, min_disparity, method)
        # return cost[:, :, :, (max_disparity - 1): (width + min_disparity)].contiguous()
        return cost

def sgm(cost, p1, p2):
    batch, max_disparity, height, width = cost.size()
    assert cost.is_contiguous()

    with torch.cuda.device_of(cost):
        cost_agg = cost.new().resize_((batch, 8, max_disparity, height, width)).zero_()
        ganet_lib.cuda_sgm(cost, cost_agg, p1, p2)
        disparity = cost_agg.sum(dim=1).argmin(dim=1)
        return disparity.float()

class SGM:
    def __init__(self, method, kernel_size, max_disparity, max_disparity_diff=1.5):
        self.max_disparity = max_disparity
        self.max_disparity_diff = max_disparity_diff
        self.kernel_size = kernel_size

        if method == 'SAD':
            self.P1 = 8 * 3 * kernel_size ** 2 / 255.0
            self.P2 = 32 * 3 * kernel_size ** 2 / 255.0
            self.cost_method = 0

        elif method == 'CENSUS_AVG':
            self.P1 = 0.1 * 3 * kernel_size ** 2
            self.P2 = 2 * 3 * kernel_size ** 2
            self.cost_method = 1

        elif method == 'CENSUS_FIX':
            self.P1 = 0.1 * 3 * kernel_size ** 2
            self.P2 = 2 * 3 * kernel_size ** 2
            self.cost_method = 2

        elif method == 'NCC':
            self.P1 = 0.1
            self.P2 = 2
            self.cost_method = 3

        else:
            raise Exception('Unknown method: ' + method)

    def process(self, X, lr_check=False):
        # left_image = (X[:, 0:3, :, :] * 255).type(torch.ByteTensor).to(X.device)
        # right_image = (X[:, 3:6, :, :] * 255).type(torch.ByteTensor).to(X.device)
        left_image = X[:, 0:3, :, :].contiguous()
        right_image = X[:, 3:6, :, :].contiguous()

        if lr_check:
            cost = calc_cost(right_image, left_image, (- self.max_disparity + 1, 1), self.kernel_size, self.cost_method)
            disp_right = sgm(cost, self.P1, self.P2) - self.max_disparity + 1
            disp_right[:, :, (1 - self.max_disparity):] = -1

            cost = calc_cost(left_image, right_image, (0, self.max_disparity), self.kernel_size, self.cost_method)
            disp_left = sgm(cost, self.P1, self.P2)
            disp_left[:, :, :(self.max_disparity - 1)] = -1

            ganet_lib.cuda_left_right_consistency_check(disp_left, disp_right, self.max_disparity_diff)
        else:
            cost = calc_cost(left_image, right_image, (0, self.max_disparity), self.kernel_size, self.cost_method)
            disp_left = sgm(cost, self.P1, self.P2)
            disp_left[:, :, :(self.max_disparity - 1)] = -1

        return cost, disp_left


class CostVolumeData:
    def __init__(self, name, cost=None, disp=None):
        self.name = name
        self.cost = cost[0].data.cpu().numpy() if cost is not None else None
        self.disp = disp[0].data.cpu().numpy() if disp is not None else None
        self.marker = None
        self.line_style = '-'
        self.line_width = 5
        self.color = None

def diff_location(input, target):
    assert input.dim() == 2
    assert target.dim() == 2

    valid_indices = (input != -1) & (target != 0)
    epe = (input - target).abs()

    epe = epe.data.cpu().numpy()
    valid_indices = valid_indices.data.cpu().numpy()

    epe_list = []
    for r in range(input.shape[0]):
        for c in range(input.shape[1]):
            if valid_indices[r, c]:
                epe_list.append((r, c, epe[r, c]))
    epe_list.sort(key=lambda x: x[2], reverse=True)
    return epe_list

def sgm_better_location(model_disp, sgm_disp, target, pixel_range=10):
    assert model_disp.dim() == 2
    assert sgm_disp.dim() == 2
    assert target.dim() == 2

    valid_indices = (model_disp != 0) & (model_disp != -1) & (sgm_disp != -1) & (target != 0)
    epe_model = (model_disp - target).abs()
    epe_sgm = (sgm_disp - target).abs()

    epe_model = epe_model.data.cpu().numpy()
    epe_sgm = epe_sgm.data.cpu().numpy()
    valid_indices = valid_indices.data.cpu().numpy()

    epe_diff = epe_model - epe_sgm

    epe_list = []
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            if valid_indices[r, c]:
                epe_list.append((r, c, epe_diff[r, c]))
    epe_list.sort(key=lambda x: x[2], reverse=True)

    sgm_better = []
    for p in epe_list:
        if not is_in_range(sgm_better, p, pixel_range):
            sgm_better.append(p)
        if len(sgm_better) >= 5:
            break

    model_better = []
    for p in epe_list[::-1]:
        if not is_in_range(model_better, p, pixel_range):
            model_better.append(p)
        if len(model_better) >= 5:
            break

    return sgm_better, model_better

def is_in_range(p_list, x, pixel_range):
    for p in p_list:
        if abs(p[0] - x[0]) <= pixel_range:
            return True
        if abs(p[1] - x[1]) <= pixel_range:
            return True
    return False


