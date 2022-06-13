import matplotlib.pyplot as plt
import tools
import utils
import os
import test.model.profile
import test.model.save_profile as save_profile
import numpy as np

class EPE_Loss:
    def __init__(self):
        self.SGM = 7.405
        self.MC_CNN = 3.79
        self.GCNet = 2.51
        self.PSMNet = 1.09
        self.GANet = 0.84
        self.GANetSmall = 6.478
        self.CSPN = 0.78

version = None
trend_kernel = 10  # version (plot) + trend kernel = real model version
epe = EPE_Loss()
profile = test.model.profile.GDFNet_mdc6f()

version, loss_history = profile.load_history(version)
print('Number of epochs:', len(loss_history['test']))

if len(loss_history['test']) == 1:
    marker = 'o'
else:
    marker = 'o'

plt.figure()
plt.title('EPE Loss')
plt.xlabel('Epoch')
plt.ylabel('EPE Loss')

p_train = plt.plot(loss_history['train'][(trend_kernel - 1):], label='Train', marker=marker)
p_test = plt.plot(loss_history['test'][(trend_kernel - 1):], label='Test', marker=marker)

test_loss_trend = np.array(loss_history['test'])
test_loss_trend = np.convolve(test_loss_trend, np.ones(trend_kernel,)) / trend_kernel
test_loss_trend = test_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
# plt.plot(test_loss_trend, label='Test Trend', marker=marker, color=p_train[0].get_color(), linestyle='--')
plt.plot(test_loss_trend, label='Test Trend', marker=marker)

train_loss_trend = np.array(loss_history['train'])
train_loss_trend = np.convolve(train_loss_trend, np.ones(trend_kernel,)) / trend_kernel
train_loss_trend = train_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
# plt.plot(train_loss_trend, label='Train Trend', marker=marker, color=p_test[0].get_color(), linestyle='--')
plt.plot(train_loss_trend, label='Train Trend', marker=marker)

plt.axhline(epe.SGM, color='b', linestyle='--', label='SGM (320 images)')
plt.axhline(epe.MC_CNN, color='g', linestyle='--', label='MC_CNN')
plt.axhline(epe.GCNet, color='r', linestyle='--', label='GCNet')
plt.axhline(epe.PSMNet, color='c', linestyle='--', label='PSMNet')
plt.axhline(epe.GANet, color='m', linestyle='--', label='GANet-15')
plt.axhline(epe.GANetSmall, color='y', linestyle='--', label='GANet-small (320 images)')
plt.axhline(epe.CSPN, color='k', linestyle='--', label='3DCSPN_ds_ss + CSPF')

plt.legend()
plt.show()