import torch.optim as optim
import torch
import torch.nn.functional as F
from utils import version_code
from utils.dataset import FlyingThings3D, random_subset, random_split, KITTI_2015, AerialImagery
from torch.utils.data import DataLoader, Subset
import os
import tools
import utils
import numpy as np
from colorama import Fore, Style
import utils.cost_volume as cv
import ganet_lib
import test.model.profile

max_disparity = 192
version = None
seed = 0
loss_threshold = 10
sgm_kernel_size = 7
lr_check = False
max_disparity_diff = 1.5
use_dir = ['left', 'right'][0]

dataset = ['flyingthings3D', 'KITTI_2015', 'KITTI_2015_benchmark', 'AerialImagery']
method = ['SAD', 'CENSUS_AVG', 'CENSUS_FIX', 'NCC']
image = ['cleanpass', 'finalpass']
pixel = [(217, 756), (56, 1037), (189, 279)]

profile = test.model.profile.GDFNet_dc6f()
dataset = dataset[1]
image = image[1]
pixel = pixel[2]

# sgms = []
# for m in method:
#     sgms.append(cv.SGM(m, sgm_kernel_size, max_disparity, 1.5))

if dataset == 'flyingthings3D':
    height = 512
    width = 960

elif dataset == 'KITTI_2015':
    # height = 352
    # width = 1216
    height, width = 336, 1200

elif dataset == 'AerialImagery':
    height, width = AerialImagery.image_size

else:
    height = None
    width = None
    raise Exception('Cannot find dataset: ' + dataset)

model = profile.load_model(max_disparity, version)[1]
print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', (height, width))
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

losses_model = []
losses_sgm = []

if model is not None:
    if dataset == 'flyingthings3D':
        test_dataset = FlyingThings3D((height, width), max_disparity, type='test', crop_seed=0, image=image,
                                      disparity=['left', 'right'])
        test_dataset = random_subset(test_dataset, 100, seed=seed)


    elif dataset == 'KITTI_2015':
        train_dataset, test_dataset = random_split(
            KITTI_2015((height, width), type='train', crop_seed=0, untexture_rate=0), seed=seed)


    elif dataset == 'KITTI_2015_benchmark':
        test_dataset = KITTI_2015((height, width), type='test', crop_seed=0)


    elif dataset == 'AerialImagery':
        test_dataset = AerialImagery()

    else:
        raise Exception('Cannot find dataset: ' + dataset)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    for batch_index, (X, Y) in enumerate(test_loader):
        # if batch_index not in [2]:
        #     continue

        if use_dir == 'right':
            X, Y = utils.flip_X(X, Y)
            Y = Y[:, 0:1, :, :]

        with torch.no_grad():
            tools.tic()
            eval_dict = profile.eval(X, Y, dataset, merge_cost=True, lr_check=False, candidate=False,
                                     regression=True,
                                     penalize=False, slope=0.2, max_disparity_diff=1.5)
            # eval_dict = profile.eval(X, Y, dataset)


            # epe_list = cv.diff_location(disp_model[0], Y[0, 0])
            # epe_list = [x for x in epe_list if x[2] < 15]
            # epe_list = epe_list[:10]
            # print(epe_list)

            # disp_sgm_model = cv.sgm(-cost_model, 0.05, 0.5)
            # mask = utils.y_mask(Y, max_disparity, dataset)
            # mask = mask & (disp_sgm_model != -1)
            # loss_sgm_model = utils.EPE_loss(Y[mask], disp_sgm_model[mask])

            time = utils.timespan_str(tools.toc(True))

            loss_str = f'loss = {utils.threshold_color(eval_dict["epe_loss"])}{eval_dict["epe_loss"]:.3f}{Style.RESET_ALL}'
            error_rate_str = f'{eval_dict["error_sum"] / eval_dict["total_eval"]:.2%}'
            print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, {error_rate_str}')
            # loss_sgm_str = f'sgm loss = {utils.threshold_color(loss_sgm_model)}{loss_sgm_model:.3f}{Style.RESET_ALL}'
            # print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, {loss_sgm_str}')

            # losses_sgm.append(float(loss_sgm_model))
            losses_model.append(float(eval_dict["epe_loss"]))

            if torch.isnan(eval_dict["epe_loss"]):
                print('detect loss nan in testing')
                exit(1)

            if str(profile) == 'GDFNet_mdc6':
                profile_name = 'GDNet'
            elif str(profile) == 'GDFNet_mdc6f':
                profile_name = 'Dual-GDNet'
            else:
                profile_name = str(profile)

            cost_volume_data = []
            if eval_dict["cost_left"] is not None:
                cv_data = cv.CostVolumeData(profile_name, - eval_dict["cost_left"])
                cv_data.line_style = '-'
                cost_volume_data.append(cv_data)

            if eval_dict["flip_cost"] is not None:
                cv_data = cv.CostVolumeData(profile_name + ' Flipped', - eval_dict["flip_cost"])
                cv_data.line_style = '-'
                cost_volume_data.append(cv_data)

            if eval_dict["cost_merge"] is not None:
                cv_data = cv.CostVolumeData(profile_name + ' Merged', - eval_dict["cost_merge"], eval_dict["disp"])
                cv_data.line_style = '-'
                cost_volume_data.append(cv_data)

            # cost_volume_data.append(cv.CostVolumeData(str(profile) + '-grad', - grad_cost, disp_model))
            # cost_volume_data.append(cv.CostVolumeData(str(profile) + '-min', None, min_disp))
            # cost_volume_data.append(cv.CostVolumeData(str(profile) + '-sgm', None, disp_sgm_model))

            # for i in range(len(method)):
            #     cost_sgm, disp_sgm = sgms[i].process(X)
            #     # valid_indices = (disp_sgm != -1) & (Y != 0)
            #     # loss_sgm = utils.EPE_loss(disp_sgm[valid_indices], Y[valid_indices])
            #     cost_sgm = F.normalize(cost_sgm, p=1, dim=1)
            #     cv_data = cv.CostVolumeData(method[i], cost_sgm, disp_sgm)
            #     cost_volume_data.append(cv_data)

            plotter = utils.CostPlotter()
            plotter.cost_volume_data = cost_volume_data
            # plotter.save_detail = {
            #     'name': str(profile),
            #     'pixel': pixel
            # }

            plotter.plot_image_disparity(X[0], Y[0, 0], dataset, eval_dict, max_disparity=max_disparity)
            # exit(0)

    print('avg model loss = {:.3f}'.format(np.array(losses_model).mean()))
    print('std model loss = {:.3f}'.format(np.array(losses_model).std()))
    # print('avg sgm loss = {:.3f}'.format(np.array(losses_sgm).mean()))
    # print('std sgm loss = {:.3f}'.format(np.array(losses_sgm).std()))


