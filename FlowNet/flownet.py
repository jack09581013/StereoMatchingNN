from keras.models import Model
from keras.layers import concatenate, Input, Conv2D, MaxPooling2D, Flatten, Deconvolution2D, LeakyReLU, BatchNormalization, Conv2DTranspose
import keras.backend as K
from keras.callbacks.callbacks import Callback
import warnings

def MSE_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def EPE_loss(y_true, y_pred):
    diff = K.square(y_pred - y_true)
    return K.mean(K.sqrt(diff[..., 0] + diff[..., 1]))

def conv(input, filters, kernel_size=3, stride=1):
    layer = Conv2D(filters, kernel_size=kernel_size, padding='same', strides=stride, use_bias=False)(input)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.1)(layer)
    return layer

def deconv(input, filters):
    layer = Deconvolution2D(filters, kernel_size=4, strides=2, padding='same', use_bias=True)(input)
    layer = LeakyReLU(alpha=0.1)(layer)
    return layer

def predict_flow(input, name):
    layer = Conv2D(2, kernel_size=3, padding='same', strides=1, use_bias=True, name=name)(input)
    return layer

def upsampling_flow(input):
    layer = Deconvolution2D(2, kernel_size=4, padding='same', strides=2, use_bias=False)(input)
    return layer

def FlowNetS():
    visible = Input(shape=(384, 512, 6))
    conv1 = conv(visible, 64, kernel_size=7, stride=2)
    conv2 = conv(conv1, 128, kernel_size=5, stride=2)
    conv3 = conv(conv2, 256, kernel_size=5, stride=2)
    conv3_1 = conv(conv3, 256)
    conv4 = conv(conv3_1, 512, stride=2)
    conv4_1 = conv(conv4, 512)
    conv5 = conv(conv4_1, 512, stride=2)
    conv5_1 = conv(conv5, 512)
    conv6 = conv(conv5_1, 1024, stride=2)

    upconv5 = deconv(conv6, 512)
    join5 = concatenate([upconv5, conv5_1])
    flow5 = predict_flow(join5, 'flow5')

    upconv4 = deconv(join5, 256)
    upconv_flow5 = upsampling_flow(flow5)
    join4 = concatenate([upconv4, conv4_1, upconv_flow5])
    flow4 = predict_flow(join4, 'flow4')

    upconv3 = deconv(join4, 128)
    upconv_flow4 = upsampling_flow(flow4)
    join3 = concatenate([upconv3, conv3_1, upconv_flow4])
    flow3 = predict_flow(join3, 'flow3')

    upconv2 = deconv(join3, 64)
    upconv_flow3 = upsampling_flow(flow3)
    join2 = concatenate([upconv2, conv2, upconv_flow3])
    flow2 = predict_flow(join2, 'flow2')

    model = Model(inputs=visible, outputs=[flow5, flow4, flow3, flow2], name='FlowNetS')

    model.compile(loss=EPE_loss,
                  optimizer='adam')

    return model

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', threshold=1, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.stop_by_threshold = False

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.threshold:
            if self.verbose > 0:
                print("Epoch {:d}: early stopping by threshold = {:.3f}, {} = {:.3f}".format(epoch, self.threshold, self.monitor, current))
            self.model.stop_training = True
            self.stop_by_threshold = True