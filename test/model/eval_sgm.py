import tools
import matplotlib.pyplot as plt
import os
from utils.dataset import RandomCropper, sub_sampling, FlyingThings3D, random_subset, KITTI_2015, random_split, AerialImagery
import utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
import utils.cost_volume as cv
import torch
from colorama import Fore, Style

max_disparity = 192
seed = 0
kernel_size = 5
dataset = ['flyingthings3D', 'KITTI_2015', 'KITTI_2015_benchmark', 'AerialImagery']
method = ['OpenCV', 'SAD', 'CENSUS_AVG', 'CENSUS_FIX', 'NCC']
image = ['cleanpass', 'finalpass']
lr_check = True
max_disparity_diff = 1.5

dataset = dataset[2]
method = method[4]
image = image[1]

print('Using dataset:', dataset)
print('Using method:', method)

if dataset == 'flyingthings3D':
    height = 512
    width = 960
    test_dataset = FlyingThings3D((height, width), max_disparity, type='test', crop_seed=0, image=image)
    test_dataset = random_subset(test_dataset, 100, seed=seed)

elif dataset == 'KITTI_2015':
    height = 352
    width = 1216
    train_dataset, test_dataset = random_split(KITTI_2015((height, width), type='train', crop_seed=0), seed=seed)

elif dataset == 'KITTI_2015_benchmark':
    height = 352
    width = 1216
    test_dataset = KITTI_2015((height, width), type='test', crop_seed=0)

elif dataset == 'AerialImagery':
    height, width = AerialImagery.image_size
    test_dataset = AerialImagery()

else:
    raise Exception('Cannot find dataset: ' + dataset)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

losses = []
error = []
total_eval = []

if method == 'OpenCV':
    # sgm = utils.SGM(max_disparity=max_disparity, mode=cv2.STEREO_SGBM_MODE_SGBM)
    sgm = utils.SGM(kernel_size, max_disparity=max_disparity, mode=cv2.STEREO_SGBM_MODE_HH)
else:
    sgm = cv.SGM(method, kernel_size, max_disparity, max_disparity_diff)

for batch_index, (X, Y) in enumerate(test_loader):
    plotter = utils.CostPlotter()
    plotter.cost_volume_data = []

    if method == 'OpenCV':
        disp = sgm.process(X)
        cost = None
    else:
        cost, disp = sgm.process(X, lr_check=lr_check)
        plotter.cost_volume_data.append(cv.CostVolumeData('cost-' + method, cost, disp))

    Y = Y[:, 0, :, :]
    mask = utils.y_mask(Y, max_disparity, dataset)
    mask = mask & (disp != -1)
    loss = utils.EPE_loss(disp[mask], Y[mask])
    error_sum = utils.error_rate(disp[mask], Y[mask], dataset)

    error.append(float(error_sum))
    total_eval.append(float(mask.float().sum()))

    eval_dict = {
        'disp': disp,
        'epe_loss': loss,
        'confidence_error': None,
        'cost_left': cost,
    }
    error_rate_str = f'{error[-1] / total_eval[-1]:.2%}'

    print(f'[{batch_index + 1}/{len(test_loader)}] loss = {utils.threshold_color(loss)}{loss:.3f}{Style.RESET_ALL}, error rate = {error_rate_str}')
    losses.append(float(loss))

    plotter.plot_image_disparity(X[0], Y[0], dataset, eval_dict,
                                 max_disparity=max_disparity,
                                 save_result_file=(f'SGM/{dataset}/{method}', batch_index, False,
                                                   error_rate_str))
    # exit(0)

print(f'Method: {method}')
print(f'Dataset: {dataset} {len(losses)} images')
print('avg loss = {:.3f}'.format(np.array(losses).mean()))
print('std loss = {:.3f}'.format(np.array(losses).std()))
print(f'avg error rates = {np.array(error).sum() / np.array(total_eval).sum():.2%}')