import pandas as pd
import numpy as np
import jacky_tool

def generate_training_data(filepath, output_filepath):
    data = pd.read_pickle(filepath)
    X = []
    flow2 = []
    flow3 = []
    flow4 = []
    flow5 = []
    folders = data['Folder'].drop_duplicates()

    for folder in folders:
        select_data = data[data['Folder'] == folder]
        for i in range(len(select_data) - 1):
            img1 = select_data.iloc[i]['ResizedImage']
            img2 = select_data.iloc[i+1]['ResizedImage']
            X += [np.concatenate([img1, img2], axis=2)]
            flow2 += [select_data.iloc[i]['Flow2']]
            flow3 += [select_data.iloc[i]['Flow3']]
            flow4 += [select_data.iloc[i]['Flow4']]
            flow5 += [select_data.iloc[i]['Flow5']]

    X = np.concatenate(X).reshape(-1, 384, 512, 6)
    flow2 = np.concatenate(flow2).reshape(-1, 96, 128, 2)
    flow3 = np.concatenate(flow3).reshape(-1, 48, 64, 2)
    flow4 = np.concatenate(flow4).reshape(-1, 24, 32, 2)
    flow5 = np.concatenate(flow5).reshape(-1, 12, 16, 2)
    Y = [flow5, flow4, flow3, flow2]

    print('X', X.shape)
    print('flow2', flow2.shape)
    print('flow3', flow3.shape)
    print('flow4', flow4.shape)
    print('flow5', flow5.shape)

    jacky_tool.save((X, Y), output_filepath)

generate_training_data('data/mpi-sintel-clean.pd', 'data/mpi-sintel-clean.np')


