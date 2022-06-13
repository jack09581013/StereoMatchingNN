from utils import *
import os
import pandas as pd

root = 'D:/MPI-Sintel-complete/training'
X = []
for folder in os.listdir(os.path.join(root, 'clean')):
    print('Process image folder:', folder)
    index = 0
    for file in os.listdir(os.path.join(root, 'clean', folder)):
        img, img_resize = get_sintel_image(os.path.join(root, 'clean', folder, file))
        X.append([folder, index, img, img_resize])
        index += 1

X = pd.DataFrame(data=X, columns=['Folder', 'Index', 'Image', 'ResizedImage'])
X.set_index(['Folder', 'Index'])

Y = []
for folder in os.listdir(os.path.join(root, 'flow')):
    print('Process flow folder:', folder)
    index = 0
    for file in os.listdir(os.path.join(root, 'flow', folder)):
        flow, flow2, flow3, flow4, flow5 = get_sintel_flow(os.path.join(root, 'flow', folder, file))
        y = np.concatenate([flow5.reshape(-1), flow4.reshape(-1), flow3.reshape(-1), flow2.reshape(-1)])
        Y.append([folder, index, y, flow, flow2, flow3, flow4, flow5])
        index += 1

Y = pd.DataFrame(data=Y, columns=['Folder', 'Index', 'FlattenFlow', 'Flow', 'Flow2', 'Flow3', 'Flow4', 'Flow5'])
Y.set_index(['Folder', 'Index'])

final = pd.merge(left=X,
                 right=Y,
                 how='left')

final.to_pickle('data/mpi-sintel-clean.pd')

# Columns = ['Folder', 'Index', 'Image', 'ResizedImage', 'FlattenFlow', 'Flow', 'Flow2', 'Flow3', 'Flow4', 'Flow5']
print(final.columns)
