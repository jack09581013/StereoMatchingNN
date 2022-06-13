import tools
import matplotlib.pyplot as plt
import os
from utils.dataset import RandomCropper, sub_sampling
from utils import plot_flying_things3D

height = 240
width = 576
ratio = 1

height = height//ratio
width = width//ratio

train_files = os.listdir('/media/jack/data/Dataset/pytorch/flyingthings3d/TRAIN')
test_files = os.listdir('/media/jack/data/Dataset/pytorch/flyingthings3d/TEST')

print('number of train files:', len(train_files))
print('number of test files:', len(test_files))

# (540, 960)
X, Y = tools.load('/media/jack/data/Dataset/pytorch/flyingthings3d/TRAIN/data_00000.np')

X, Y = sub_sampling(X, Y, ratio)

cropper = RandomCropper(X.shape[1:3], (height, width), seed=0)
X, Y = cropper.crop(X), cropper.crop(Y)

plot_flying_things3D(X, Y, None)

