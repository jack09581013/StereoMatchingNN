import torch.optim as optim
import torch
import torch.nn.functional as F
from GANet.GANet import GANet
from GANet.GANet_small import GANetSmall
from utils import version_code
from utils.dataset import FlyingThings3D, random_subset, random_split, KITTI_2015
from torch.utils.data import DataLoader, Subset
import os
import tools
import utils
import cv2
import matplotlib.pyplot as plt
import test.model.profile as profile
import test

max_disparity = 192
version = 384
seed = 0

dataset = ['flyingthings3D', 'KITTI_2015']
image = ['cleanpass', 'finalpass']

profile = profile.GDFNet_mdc6()
dataset = dataset[0]
image = image[1]

save_root = '../../result/{}/{}'.format(profile, dataset)

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

sgm = utils.SGM(max_disparity=max_disparity, mode=cv2.STEREO_SGBM_MODE_SGBM)

model = profile.load_model(max_disparity, version)[1]
version, loss_history = profile.load_history(version)

print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', (height, width))
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

if model is not None:
    if dataset == 'flyingthings3D':
        test_dataset = FlyingThings3D((height, width), max_disparity, type='test', crop_seed=0, image=image)
        test_dataset = random_subset(test_dataset, 40, seed=seed)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    elif dataset == 'KITTI_2015':
        train_dataset, test_dataset = random_split(KITTI_2015((height, width), type='train', crop_seed=0), seed=seed)
        # test_dataset = KITTI_2015((height, width), type='test', crop_seed=0)
        # test_dataset = random_subset(test_dataset, 100, seed=seed)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    else:
        raise Exception('Cannot find dataset: ' + dataset)

    model.eval()
    for batch_index, (X, Y) in enumerate(test_loader):
        with torch.no_grad():
            if isinstance(profile, test.model.profile.GDFNet_mdc6f):
                eval_dict = profile.eval(X, Y, dataset, lr_check=False, candidate=True, regression=True)

            elif isinstance(profile, test.model.profile.GDFNet_mdc6):
                eval_dict = profile.eval(X, Y, dataset, lr_check=False, candidate=False, regression=True)
                # eval_dict = profile.eval(X, Y, dataset, lr_check=True, regression=True, penalize=False, slope=1,
                #                          max_disparity_diff=1.5)
            else:
                eval_dict = profile.eval(X, Y, dataset)
            # utils.plot_image_disparity(X[0], Y[0], disp_model[0], loss_model)
            # exit(0)

            if torch.isnan(eval_dict['epe_loss']):
                print('detect loss nan in testing')
                exit(1)

        disp_sgm = sgm.process(X)
        Y = Y[:, 0, :, :]
        mask = utils.y_mask(Y, max_disparity, dataset)
        disp_sgm[:, :, :max_disparity] = -1
        mask_sgm = mask & (disp_sgm != -1)
        sgm_loss = utils.EPE_loss(disp_sgm[mask_sgm], Y[mask_sgm])

        print('[{}/{}]'.format(batch_index + 1, len(test_loader)))
        print('\tmodel_loss = {:.3f}'.format(eval_dict['epe_loss']))
        print('\tsgm_loss = {:.3f}'.format(sgm_loss))

        utils.save_comparision(X[0], Y[0], eval_dict['disp'][0], disp_sgm[0], eval_dict['epe_loss'], sgm_loss, save_root,
                               'S{:03d}-B{:03d}.png'.format(seed, batch_index), str(profile))
