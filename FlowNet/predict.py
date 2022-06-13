from utils import *
import jacky_tool
from keras.models import load_model
import os
from flownet import *

def EPE_loss_np(y_true, y_pred):
    diff = np.square(y_pred - y_true)
    return np.mean(np.sqrt(diff[..., 0] + diff[..., 1]))


def predict_model(model, X, Y, flow_num=2):
    flows = model.predict(X)

    for i in range(X.shape[0]):
        plt.subplot(X.shape[0], 3, i * 3 + 1)
        plt.title('Image')
        plt.imshow(X[i, ..., 0:3])

        plt.subplot(X.shape[0], 3, i * 3 + 2)
        plt.title('Ground True')
        plt.imshow(xy_to_color(Y[5 - flow_num][i]))

        loss = EPE_loss_np(Y[5 - flow_num][i], flows[5 - flow_num][i])

        plt.subplot(X.shape[0], 3, i * 3 + 3)
        plt.title('Predict (EPE = {:.3f})'.format(loss))
        plt.imshow(xy_to_color(flows[5 - flow_num][i]))


dataset = ['mpi-sintel-clean.np', 'flying_chairs_00001_02000.np', 'flying_chairs_02001_04000.np']
X, Y = jacky_tool.load(os.path.join(r'D:\Dataset\python', dataset[0]))
X = X / 255.0

model = load_model('model/FlowNetS.nn', custom_objects={'EPE_loss': EPE_loss})
model.summary()

for start in range(0, 200, 50):
    end = start + 1
    fig = plt.figure()
    fig.canvas.set_window_title('Start = {}'.format(start))
    predict_model(model, X[start:end], [y[start:end] for y in Y], flow_num=2)
plt.show()

