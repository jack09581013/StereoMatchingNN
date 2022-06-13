from torch.utils.data import Dataset, Subset
import torch
import tools
import os
from utils import *
import cv2
import random

class FlyingThings3D(Dataset):
    # ROOT = '/media/jack/data/Dataset/pytorch/flyingthings3d'
    ROOT = r'D:\Dataset\pytorch\flyingthings3d'

    def __init__(self, crop_size, max_disparity, type='train', image='cleanpass', crop_seed=None, down_sampling=1,
                 disparity=['left'], small=False):
        if small:
            self.ROOT += '_s'

        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        assert isinstance(down_sampling, int)
        self.down_sampling = down_sampling
        self.disparity = disparity
        self.data_max_disparity = []

        if type == 'train':
            for d in self.disparity:
                self.data_max_disparity.append(tools.load(os.path.join(self.ROOT, f'{d}_max_disparity.np'))[0])
            self.root = os.path.join(self.ROOT, 'TRAIN')

            if small:
                self.size = 7460
            else:
                self.size = 22390
        elif type == 'test':
            for d in self.disparity:
                self.data_max_disparity.append(tools.load(os.path.join(self.ROOT, f'{d}_max_disparity.np'))[1])
            self.root = os.path.join(self.ROOT, 'TEST')

            if small:
                self.size = 1440
            else:
                self.size = 4370

        else:
            raise Exception(f'Unknown type: "{type}"')

        self.mask = np.ones(self.size, dtype=np.uint8)
        for d in self.data_max_disparity:
            self.mask = self.mask & (d < max_disparity - 1)
        self.size = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask)

        if image not in ['cleanpass', 'finalpass']:
            raise Exception(f'Unknown image: "{image}"')

        self.image = image
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self._make_mask_index()

    def __getitem__(self, index):
        index = self.mask_index[index]
        X = tools.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))
        cropper = RandomCropper(X.shape[1:3], self.crop_size, seed=self.crop_seed)
        X = torch.from_numpy(X)
        X = cropper.crop(X).float() / 255

        Y_list = []
        for d in self.disparity:
            Y = tools.load(os.path.join(self.root, f'{d}_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y = cropper.crop(Y).unsqueeze(0)
            Y_list.append(Y)
        Y = torch.cat(Y_list, dim=0)

        if self.down_sampling != 1:
            X = X[:, ::self.down_sampling, ::self.down_sampling]
            Y = Y[:, ::self.down_sampling, ::self.down_sampling]
            Y /= self.down_sampling

        return X.cuda(), Y.cuda()

    def __len__(self):
        return self.size

    def _make_mask_index(self):
        self.mask_index = np.zeros(self.size, dtype=np.int)

        i = 0
        m = 0
        while i < len(self.mask):
            if self.mask[i]:
                self.mask_index[m] = i
                m += 1
            i += 1

class KITTI_2015(Dataset):
    ROOT = r'G:\Dataset\KITTI 2015'

    def __init__(self, crop_size, type='train', crop_seed=None, untexture_rate=0.1):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        self.type = type
        if type == 'train':
            self.root = os.path.join(self.ROOT, 'training')
        elif type == 'test':
            self.root = os.path.join(self.ROOT, 'testing')
        else:
            raise Exception('Unknown type "{}"'.format(type))

        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self.untexture_rate = untexture_rate

    def __getitem__(self, index):
        if self.type == 'train':
            untexture_learning = random.randint(1, 100) <= int(self.untexture_rate*100)
            if untexture_learning:
                print('untexture_learning')
                bgr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                X1 = np.full((375, 1242, 3), bgr, dtype=np.uint8)
                X2 = np.full((375, 1242, 3), bgr, dtype=np.uint8)
                Y = np.full((375, 1242), 0.001, dtype=np.float32)

                cropper = RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                X = np.concatenate([X1, X2], axis=2)
                X = X.swapaxes(0, 2).swapaxes(1, 2)
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)
                X, Y = cropper.crop(X).float() / 255, cropper.crop(Y).float()
                return X.cuda(), Y.cuda()
            else:
                X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                Y = cv2.imread(os.path.join(self.root, 'disp_occ_0/{:06d}_10.png'.format(index)))

                cropper = RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)

                X1 = rgb2bgr(X1)
                X2 = rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)
                X = X.swapaxes(0, 2).swapaxes(1, 2)
                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)
                X, Y = cropper.crop(X).float() / 255, cropper.crop(Y).float()
                return X.cuda(), Y.cuda()

        elif self.type == 'test':
            X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
            X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
            cropper = RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)

            X1 = rgb2bgr(X1)
            X2 = rgb2bgr(X2)

            X = np.concatenate([X1, X2], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)
            X = torch.from_numpy(X)
            X = cropper.crop(X) / 255.0

            Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)

            return X.cuda(), Y.cuda()

    def __len__(self):
        if self.type == 'train':
            return 200
        if self.type == 'test':
            return 20

class KITTI_2015_benchmark(Dataset):
    ROOT = r'D:\Dataset\KITTI 2015'
    # HEIGHT, WIDTH = 384, 1248
    HEIGHT, WIDTH = 352, 1216

    def __init__(self):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        self.root = os.path.join(self.ROOT, 'testing')

    def __getitem__(self, index):
        X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
        X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
        origin_height, origin_width = X1.shape[:2]

        # Ground true (375, 1242, 3), dtype uint8
        # print(cv2.imread(f'D:/Dataset/KITTI 2015/training/disp_noc_0/{index:06d}_10.png')[250:, 250:])

        X1 = cv2.resize(X1, (self.WIDTH, self.HEIGHT))
        X2 = cv2.resize(X2, (self.WIDTH, self.HEIGHT))

        X1 = rgb2bgr(X1)
        X2 = rgb2bgr(X2)

        X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
        X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width
        X = torch.from_numpy(X) / 255.0

        Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)
        return X.cuda(), Y.cuda(), origin_height, origin_width

    def __len__(self):
        return 200

class AerialImagery(Dataset):
    ROOT = '/media/jack/data/Dataset/aerial imagery'
    # image_size = (800, 1280)
    image_size = (384, 1280)

    def __init__(self):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'

        self.rc = [(3020, 3015), (3200, 4500), (2950, 5760)][-2:-1]
        self.disp = 400

    def __getitem__(self, index):
        r, c = self.rc[index]
        os.makedirs(os.path.join(self.ROOT, 'cache'), exist_ok=True)
        cach_path = os.path.join(self.ROOT, 'cache', f'{r:d}_{c:d}_{self.image_size[0]}x{self.image_size[1]}.np')

        if os.path.exists(cach_path):
            print(f'using cache: {cach_path}')
            X = tools.load(cach_path)

        else:
            left_image = cv2.imread(os.path.join(self.ROOT, '0-rectL.tif'))
            right_image = cv2.imread(os.path.join(self.ROOT, '0-rectR.tif'))

            left_image = cv2.rotate(left_image, cv2.ROTATE_90_CLOCKWISE)
            right_image = cv2.rotate(right_image, cv2.ROTATE_90_CLOCKWISE)

            left_image = rgb2bgr(left_image)
            right_image = rgb2bgr(right_image)

            height, width = self.image_size
            left_image = left_image[r:r + height, c:c + width]
            right_image = right_image[r:r + height, c - self.disp:c - self.disp + width]

            X = np.concatenate([left_image, right_image], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)
            X = torch.from_numpy(X)
            tools.save(X, cach_path)

        Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)

        return X.cuda() / 255.0, Y.cuda()

    def __len__(self):
        return len(self.rc)

def random_subset(dataset, size, seed=None):
    assert size <= len(dataset), 'subset size cannot larger than dataset'
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = indexes[:size]
    return Subset(dataset, indexes)

def random_split(dataset, train_ratio=0.8, seed=None):
    assert 0 <= train_ratio <= 1
    train_size = int(train_ratio * len(dataset))
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_indexes = indexes[:train_size]
    test_indexes = indexes[train_size:]
    return Subset(dataset, train_indexes), Subset(dataset, test_indexes)

def sub_sampling(X, Y, ratio):
    X = X[:, ::ratio, ::ratio]
    Y = Y[::ratio, ::ratio] / ratio
    return X, Y
