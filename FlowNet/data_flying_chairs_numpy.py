import pandas as pd
import numpy as np
import jacky_tool
import os
from utils import *

root = r'D:\FlyingChairs\FlyingChairs_release\data'
step = 2000
for start in range(1, 22873, step):
    X = []
    flow2_list = []
    flow3_list = []
    flow4_list = []
    flow5_list = []

    end = start + step if start + step < 22873 else 22873
    for i in range(start, end):
        print('Process:', i)
        img1 = get_ppm(os.path.join(root, '{:05d}_img1.ppm'.format(i)))
        img2 = get_ppm(os.path.join(root, '{:05d}_img2.ppm'.format(i)))
        flow, flow2, flow3, flow4, flow5 = get_flying_chairs_flow(os.path.join(root, '{:05d}_flow.flo'.format(i)))

        X.append(np.concatenate([img1, img2], axis=2))
        flow2_list.append(flow2)
        flow3_list.append(flow3)
        flow4_list.append(flow4)
        flow5_list.append(flow5)

    X = np.concatenate(X).reshape(-1, 384, 512, 6)
    flow2_list = np.concatenate(flow2_list).reshape(-1, 96, 128, 2)
    flow3_list = np.concatenate(flow3_list).reshape(-1, 48, 64, 2)
    flow4_list = np.concatenate(flow4_list).reshape(-1, 24, 32, 2)
    flow5_list = np.concatenate(flow5_list).reshape(-1, 12, 16, 2)
    Y = [flow5_list, flow4_list, flow3_list, flow2_list]

    print('X', X.shape)
    print('flow2', flow2_list.shape)
    print('flow3', flow3_list.shape)
    print('flow4', flow4_list.shape)
    print('flow5', flow5_list.shape)

    jacky_tool.save((X, Y), 'data/flying_chairs_{:05d}_{:05d}.np'.format(start, end-1))


