import torch.optim as optim
import torch
import torch.nn.functional as F
from utils.dataset import FlyingThings3D, random_subset, random_split, KITTI_2015
from torch.utils.data import DataLoader, Subset
import os
import tools
import utils
from test.model.profile import *
from colorama import Fore, Style
import test.model.profile
import numpy as np

# height, width = 240, 576 GDFNet_mdc6f
# height, width = 256, 512
height, width = 192, 384
max_disparity = 144
# max_disparity = 192
version = None
max_version = 750
batch = 1
seed = 0
loss_threshold = 10
full_dataset = True
small_dataset = False
untexture_rate = 0
dataset = ['flyingthings3D', 'KITTI_2015']
image = ['cleanpass', 'finalpass']  # for flyingthings3D

profile = test.model.profile.GDFNet_mdc6f()
dataset = dataset[1]
image = image[1]

model = profile.load_model(max_disparity, version)[1]
version, loss_history = profile.load_history(version)

print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', (height, width))
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

if dataset == 'flyingthings3D':
    train_dataset = FlyingThings3D((height, width), max_disparity, type='train', crop_seed=None, image=image, small=small_dataset)
    test_dataset = FlyingThings3D((height, width), max_disparity, type='test', crop_seed=None, small=small_dataset)

    if not full_dataset:
        train_dataset = random_subset(train_dataset, 1920, seed=seed)
        test_dataset = random_subset(test_dataset, 480, seed=seed)

elif dataset == 'KITTI_2015':
    train_dataset, test_dataset = random_split(KITTI_2015((height, width), type='train', crop_seed=None,
                                                          untexture_rate=untexture_rate), seed=seed)
else:
    raise Exception('Cannot find dataset: ' + dataset)

print('Number of training data:', len(train_dataset))
print('Number of testing data:', len(test_dataset))
os.system('nvidia-smi')

# 5235 MB
for v in range(version, max_version + 1):
    if dataset == 'flyingthings3D':
        train_loader = DataLoader(random_subset(train_dataset, 192), batch_size=batch, shuffle=False)
        test_loader = DataLoader(random_subset(test_dataset, 48), batch_size=batch, shuffle=False)

    elif dataset == 'KITTI_2015':
        train_loader = DataLoader(random_subset(train_dataset, 160), batch_size=batch, shuffle=False)
        test_loader = DataLoader(random_subset(test_dataset, 40), batch_size=batch, shuffle=False)
    else:
        raise Exception('Cannot find dataset: ' + dataset)

    train_loss = []
    test_loss = []
    error = []
    total_eval = []

    print('Start training, version = {}'.format(v))
    model.train()
    for batch_index, (X, Y) in enumerate(train_loader):
        tools.tic()

        if isinstance(profile, test.model.profile.GDFNet_mdc6f):
            optimizer.zero_grad()
            train_dict0 = profile.train(X, Y, dataset, flip=False)
            train_dict0['loss'].backward()
            optimizer.step()

            optimizer.zero_grad()
            train_dict1 = profile.train(X, Y, dataset, flip=True)
            train_dict1['loss'].backward()
            optimizer.step()

            wl = width / (2 * width - max_disparity)
            wr = (width - max_disparity) / (2 * width - max_disparity)

            loss = wl*train_dict0['loss'] + wr*train_dict1['loss']
            epe_loss = wl*train_dict0['epe_loss'] + wr*train_dict1['epe_loss']
        else:
            optimizer.zero_grad()
            train_dict = profile.train(X, Y, dataset)
            train_dict['loss'].backward()
            loss = train_dict['loss']
            epe_loss = train_dict['epe_loss']
            optimizer.step()

        train_loss.append(float(epe_loss))

        time = utils.timespan_str(tools.toc(True))
        loss_str = f'loss = {utils.threshold_color(loss)}{loss:.3f}{Style.RESET_ALL}'
        epe_loss_str = f'epe_loss = {utils.threshold_color(epe_loss)}{epe_loss:.3f}{Style.RESET_ALL}'
        print(f'[{batch_index + 1}/{len(train_loader)} {time}] {loss_str}, {epe_loss_str}')

        # plotter = utils.CostPlotter()
        # plotter.plot_image_disparity(X[0], Y[0, 0], dataset, train_dict,
        #                              max_disparity=max_disparity)

        if torch.isnan(loss):
            print('detect loss nan in training')
            exit(0)

    train_loss = float(torch.tensor(train_loss).mean())
    print(f'Avg train loss = {utils.threshold_color(train_loss)}{train_loss:.3f}{Style.RESET_ALL}')

    print('Start testing, version = {}'.format(v))
    model.eval()
    for batch_index, (X, Y) in enumerate(test_loader):
        with torch.no_grad():
            tools.tic()
            if isinstance(profile, test.model.profile.GDFNet_mdc6):
                eval_dict = profile.eval(X, Y, dataset, merge_cost=False, lr_check=False, candidate=False,
                                         regression=True,
                                         penalize=False, slope=1, max_disparity_diff=1.5)
            else:
                eval_dict = profile.eval(X, Y, dataset)

            time = utils.timespan_str(tools.toc(True))
            loss_str = f'epe loss = {utils.threshold_color(eval_dict["epe_loss"])}{eval_dict["epe_loss"]:.3f}{Style.RESET_ALL}'
            error_rate_str = f'error rate = {eval_dict["error_sum"] / eval_dict["total_eval"]:.2%}'
            print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, {error_rate_str}')

            test_loss.append(float(eval_dict["epe_loss"]))
            error.append(float(eval_dict["error_sum"]))
            total_eval.append(float(eval_dict["total_eval"]))

            # plotter = utils.CostPlotter()
            # plotter.plot_image_disparity(X[0], Y[0, 0], dataset, eval_dict,
            #                              max_disparity=max_disparity)

            if torch.isnan(eval_dict["epe_loss"]):
                print('detect loss nan in testing')
                exit(0)

    test_loss = float(torch.tensor(test_loss).mean())
    test_error_rate = np.array(error).sum() / np.array(total_eval).sum()
    loss_str = f'epe loss = {utils.threshold_color(test_loss)}{test_loss:.3f}{Style.RESET_ALL}'
    error_rate_str = f'error rate = {test_error_rate:.2%}'
    print(f'Avg {loss_str}, {error_rate_str}')

    loss_history['train'].append(train_loss)
    loss_history['test'].append(test_loss)

    print('Start save model')
    profile.save_version(model, loss_history, v)
