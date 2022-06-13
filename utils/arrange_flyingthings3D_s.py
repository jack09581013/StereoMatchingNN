import cv2
import os
import tools
import numpy as np
import utils
dataset_root = '/media/jack/data/Dataset'
# use_image = ['cleanpass', 'finalpass'][0]

os.chdir(dataset_root)
print('dataset root on', os.getcwd())

for f1 in ['TRAIN', 'TEST']:
    save_path = os.path.join('pytorch/flyingthings3d_s', f1)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'cleanpass'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'finalpass'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'left_disparity'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'right_disparity'), exist_ok=True)
    index = 0
    f2 = 'A'
    subfolder = f1 + '/' + f2

    cleanpass_root = os.path.join(f'flyingthings3d/frames_cleanpass/', subfolder)
    finalpass_root = os.path.join(f'flyingthings3d/frames_finalpass/', subfolder)
    disparity_root = os.path.join('flyingthings3d/disparity/', subfolder)

    for folder in os.listdir(cleanpass_root):
        for file in os.listdir(os.path.join(cleanpass_root, folder, 'left')):
            print('process [{}/{}] {}'.format(subfolder, folder, index))

            # Clean pass
            left_image = cv2.imread(os.path.join(cleanpass_root, folder, 'left', file))
            right_image = cv2.imread(os.path.join(cleanpass_root, folder, 'right', file))
            X = np.concatenate([left_image, right_image], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)
            tools.save(X, os.path.join(save_path, f'cleanpass/{index:05d}.np'))

            # Final pass
            left_image = cv2.imread(os.path.join(finalpass_root, folder, 'left', file))
            right_image = cv2.imread(os.path.join(finalpass_root, folder, 'right', file))
            X = np.concatenate([left_image, right_image], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)
            tools.save(X, os.path.join(save_path, f'finalpass/{index:05d}.np'))

            # Left Disparity
            Y = utils.read_pfm(os.path.join(disparity_root, folder, 'left', file[:4] + '.pfm')).squeeze()
            tools.save(Y, os.path.join(save_path, f'left_disparity/{index:05d}.np'))

            # Right Disparity
            Y = utils.read_pfm(os.path.join(disparity_root, folder, 'right', file[:4] + '.pfm')).squeeze()
            tools.save(Y, os.path.join(save_path, f'right_disparity/{index:05d}.np'))

            index += 1



