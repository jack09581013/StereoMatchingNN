from torch.utils.data import Dataset, Subset
import torch
import tools
import os
from utils import *
import cv2
from utils.dataset import FlyingThings3D, random_subset
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from colorama import Fore, Style

small = True
ROOT = '/media/jack/data/Dataset/pytorch'

if small:
    ROOT = os.path.join(ROOT, 'flyingthings3d_s')
    meta = [('TRAIN', 7460), ('TEST', 1440)]
else:
    ROOT = os.path.join(ROOT, 'flyingthings3d')
    meta = [('TRAIN', 22390), ('TEST', 4370)]

left_max_disparity = {
    'TRAIN': [],
    'TEST': []
}

right_max_disparity = {
    'TRAIN': [],
    'TEST': []
}

for data_type, data_size in meta:
    for index in range(data_size):
        Y = tools.load(os.path.join(ROOT, data_type, f'left_disparity/{index:05d}.np'))
        Y_max_disp_L = Y.reshape(-1).max()

        Y = tools.load(os.path.join(ROOT, data_type, f'right_disparity/{index:05d}.np'))
        Y_max_disp_R = Y.reshape(-1).max()

        print(f'[{index+1}/{data_size} {data_type}] {Y_max_disp_L:7.3f} {Y_max_disp_R:7.3f}')

        left_max_disparity[data_type].append(Y_max_disp_L)
        right_max_disparity[data_type].append(Y_max_disp_R)

left_max_disparity['TRAIN'] = np.array(left_max_disparity['TRAIN'])
left_max_disparity['TEST'] = np.array(left_max_disparity['TEST'])

right_max_disparity['TRAIN'] = np.array(right_max_disparity['TRAIN'])
right_max_disparity['TEST'] = np.array(right_max_disparity['TEST'])

left_max_disparity = (left_max_disparity['TRAIN'], left_max_disparity['TEST'])
right_max_disparity = (right_max_disparity['TRAIN'], right_max_disparity['TEST'])

tools.save(left_max_disparity, os.path.join(ROOT, 'left_max_disparity.np'))
tools.save(right_max_disparity, os.path.join(ROOT, 'right_max_disparity.np'))