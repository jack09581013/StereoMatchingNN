import torch.optim as optim
import torch
import torch.nn.functional as F
from utils import version_code
from utils.dataset import FlyingThings3D, random_subset, KITTI_2015
from torch.utils.data import DataLoader, random_split, Subset
import os
import tools
import utils
import numpy as np
from test.model.profile import *
from colorama import Fore, Style
import utils.cost_volume as cv
import ganet_lib

max_disparity = 128
version = None
seed = 0
loss_threshold = 10
sgm_kernel_size = 7

dataset = ['flyingthings3D', 'KITTI_2015'][1]
method = 'NCC'
profile = GANet_deep_Profile(max_disparity)

if dataset == 'flyingthings3D':
    height = int(256 * 1.5)
    width = int(512 * 1.5)

elif dataset == 'KITTI_2015':
    height = int(256)
    width = int(512 * 2)
else:
    height = None
    width = None
    raise Exception('Cannot find dataset: ' + dataset)

version, model, loss_history = profile.load_version(version)
sgm = cv.SGM(method, sgm_kernel_size, max_disparity, 1.5)

print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', (height, width))
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

if model is not None:
    if dataset == 'flyingthings3D':
        test_dataset = FlyingThings3D((height, width), type='test', crop_seed=0, image='cleanpass')
        test_dataset = random_subset(test_dataset, 4, seed=seed)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    elif dataset == 'KITTI_2015':
        test_dataset = KITTI_2015((height, width), type='train', crop_seed=0)
        test_dataset = random_subset(test_dataset, 4, seed=seed)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        raise Exception('Cannot find dataset: ' + dataset)

    model.eval()
    for batch_index, (X, Y) in enumerate(test_loader):
        Y_max_disp = Y.view(-1).max()
        if Y_max_disp >= max_disparity:
            print(f'{Fore.RED}[{batch_index + 1}/{len(test_loader)}] detect large disparity = {Y_max_disp:.3f}{Style.RESET_ALL}')
            continue

        with torch.no_grad():
            # cost_model, disp_model = cv.model_left_right_consistency_check(model, X, 1.5)
            cost0_model, cost1_model, disp_model = model(X[:, 0:3, :, :], X[:, 3:6, :, :])
            cost_sgm, disp_sgm = sgm.process(X)

            sgm_better, model_better = cv.sgm_better_location(disp_model[0], disp_sgm[0], Y[0])

            mask = utils.y_mask(Y, max_disparity, dataset)
            mask_model = mask & (disp_model != -1)
            mask_sgm = mask & (disp_sgm != -1)

            loss_model = utils.EPE_loss(Y[mask_model], disp_model[mask_model])
            loss_sgm = utils.EPE_loss(Y[mask_sgm], disp_sgm[mask_sgm])

            print(f'[{batch_index + 1}/{len(test_loader)}]')
            print(f'\tmodel loss = {Fore.GREEN if loss_model < loss_threshold else Fore.RED}{loss_model:.3f}{Style.RESET_ALL}')
            print(f'\tncc loss = {Fore.GREEN if loss_sgm < loss_threshold else Fore.RED}{loss_sgm:.3f}{Style.RESET_ALL}')

        cost_sgm = F.normalize(cost_sgm, p=1, dim=1)
        cost0_model = F.normalize(cost0_model, p=1, dim=1)

        cost_volume_data = []
        cost_volume_data.append(cv.CostVolumeData('GANet-deep-0', -cost0_model, disp_model))
        cost_volume_data.append(cv.CostVolumeData('GANet-deep-1', -cost1_model, disp_model))
        cost_volume_data.append(cv.CostVolumeData(method, cost_sgm, disp_sgm))

        sub_folders = ['sgm-better', 'model-better']
        for sub_list, sub_folder in zip([sgm_better, model_better], sub_folders):
            root = os.path.join(f'../../result/Cost Volume/{dataset}/{batch_index+1}', sub_folder)
            for i, location in enumerate(sub_list):
                sub_root = os.path.join(root, str(i + 1))
                os.makedirs(sub_root, exist_ok=True)
                utils.plot_image_disparity(X[0], Y[0], disp_model[0], loss_model, location=location[0:2],
                                           max_disparity=max_disparity,
                                           save_file=os.path.join(sub_root, f'model.png'))

                utils.plot_image_disparity(X[0], Y[0], disp_sgm[0], loss_sgm, location=location[0:2],
                                           max_disparity=max_disparity,
                                           save_file=os.path.join(sub_root, f'sgm.png'))

                utils.plot_cost_volume(location[0:2], cost_volume_data, Y,
                                       save_file=os.path.join(sub_root, f'cost_volume.png'))




