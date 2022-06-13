from flownet import *
import jacky_tool
from keras.models import load_model
import os
import random
import winsound
import numpy as np

def train_model(X, Y, stop_callback, model_func=FlowNetS, overwrite=False, summary=False, epochs=10, batch_size=16):
    if overwrite:
        model = model_func()
    else:
        if os.path.exists('model/{}.nn'.format(model_func.__name__)):
            model = load_model('model/{}.nn'.format(model_func.__name__), custom_objects={'EPE_loss': EPE_loss})
        else:
            model = model_func()

    if summary:
        print(model.summary())

    history = model.fit(X, Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[stop_callback]).history

    history_file_path = 'model/{}.ht'.format(model_func.__name__)
    if overwrite:
        history = {
            'loss': history['loss'],
            'val_loss': history['val_loss']
        }
        jacky_tool.save(history, history_file_path)

    else:
        if os.path.exists(history_file_path):
            origin_history = jacky_tool.load(history_file_path)
            origin_history['loss'].extend(history['loss'])
            origin_history['val_loss'].extend(history['val_loss'])
            jacky_tool.save(origin_history, history_file_path)
        else:
            history = {
                'loss': history['loss'],
                'val_loss': history['val_loss']
            }
            jacky_tool.save(history, history_file_path)

    model.save('model/{}.nn'.format(model_func.__name__))

def test_train(data_file_path):
    stop_callback = EarlyStoppingByLossVal(monitor='val_loss', threshold=1)
    X, Y = jacky_tool.load(data_file_path)
    X = X[:10]
    Y = [y[:10] for y in Y]
    train_model(X / 255.0, Y, stop_callback, overwrite=True, summary=True, epochs=50, batch_size=1)

def test_train_2000():
    stop_callback = EarlyStoppingByLossVal(monitor='val_loss', threshold=1)
    X, Y = jacky_tool.load('data/flying_chairs_{:05d}_{:05d}.np'.format(1, 2000))
    train_model(X / 255.0, Y,stop_callback, overwrite=True, summary=True, epochs=15, batch_size=16)
    winsound.Beep(440, 1000)

def full_train(round_limit=20, stop_threshold=5):
    """Using flying chair to train data"""
    chair_pairs = []
    step = 2000
    for start in range(1, 22873, step):
        end = start + step if start + step < 22873 else 22873
        chair_pairs.append((start, end - 1))

    indexes = np.arange(0, len(chair_pairs))

    rounds = 0
    stop_by_threshold = False

    while not stop_by_threshold and rounds < round_limit:
        rounds += 1
        np.random.shuffle(indexes)
        stop_by_threshold = train_one_round(indexes, chair_pairs, stop_threshold)

    print('Rounds = {}'.format(rounds))
    winsound.Beep(440, 1000)

def train_one_round(indexes, chair_pairs, stop_threshold):
    for i in np.nditer(indexes):
        pair = chair_pairs[i]
        print('Use flying chairs pair:', pair)
        X, Y = jacky_tool.load(os.path.join(r'D:\Dataset\python', 'flying_chairs_{:05d}_{:05d}.np'.format(*pair)))
        X = X / 255.0
        stop_callback = EarlyStoppingByLossVal(monitor='val_loss', threshold=stop_threshold)
        train_model(X, Y, stop_callback, overwrite=False, summary=False, epochs=10, batch_size=16)

        if stop_callback.stop_by_threshold:
            return True
    return False

dataset = ['mpi-sintel-clean.np', 'flying_chairs_00001_02000.np']

# test_train(os.path.join('data', dataset[0]))
# test_train_2000()
full_train(round_limit=50, stop_threshold=2)







