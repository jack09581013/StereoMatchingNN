import numpy as np
import matplotlib.pyplot as plt
import struct
import cv2

def angle_to_hue(angle):
    return np.array(angle / 2, dtype=np.uint8)

def radian_to_hue(radian):
    return np.array(radian / np.pi * 90, dtype=np.uint8)

# Hue = 0 ~ 180 (uint8)
def xy_to_color(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = radian_to_hue(ang)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def plot_flow_color_line():
    hsv = np.zeros((1, 360, 3), dtype=np.uint8)
    hsv[..., 0] = angle_to_hue(np.array([ang for ang in range(0, 360, 1)]))
    hsv[..., 1:3] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.yticks([])
    plt.imshow(rgb)
    plt.show()

def get_flo(filepath):
    """
        https://towardsdatascience.com/generating-optical-flow-using-nvidia-flownet2-pytorch-implementation-d7b0ae6f8320
        ".flo" file format used for optical flow evaluation
        Stores 2-band float image for horizontal (u) and vertical (v) flow components.
        Floats are stored in little-endian order.
        A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
          bytes  contents
          0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
                  (just a sanity check that floats are represented correctly)
          4-7     width as an integer
          8-11    height as an integer
          12-end  data (width*height*2*4 bytes total)
                  the float values for u and v, interleaved, in row order, i.e.,
                  u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
    """
    with open(filepath, 'rb') as flo:
        tag = flo.read(4)
        width = struct.unpack('<i', flo.read(4))
        height = struct.unpack('<i', flo.read(4))
        flow = np.frombuffer(flo.read(), dtype=np.float32)
        flow.shape = (height[0], width[0], 2)
    return flow

def get_ppm(filepath):
    image = cv2.imread(filepath)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# resize = (width, height)
def get_sintel_flow(filepath, resize=(128, 96)):
    flow = get_flo(filepath)
    flow2 = cv2.resize(flow, resize)
    flow3 = cv2.resize(flow2, divide_shape(flow2.shape, 2))
    flow4 = cv2.resize(flow2, divide_shape(flow2.shape, 4))
    flow5 = cv2.resize(flow2, divide_shape(flow2.shape, 8))
    return flow, flow2, flow3, flow4, flow5

def get_sintel_image(filepath, resize=(512, 384)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, resize)
    return img, img_resize

# resize = (width, height)
def get_flying_chairs_flow(filepath, resize=(128, 96)):
    flow = get_flo(filepath)
    flow2 = cv2.resize(flow, resize)
    flow3 = cv2.resize(flow2, divide_shape(flow2.shape, 2))
    flow4 = cv2.resize(flow2, divide_shape(flow2.shape, 4))
    flow5 = cv2.resize(flow2, divide_shape(flow2.shape, 8))
    return flow, flow2, flow3, flow4, flow5

def divide_shape(shape, x):
    return int(shape[1]/x), int(shape[0]/x)

def seperate_prediction(prediction):
    """prediction's size is 32640"""
    # Flow5, Flow4, Flow3, Flow2
    size = [384, 1536, 6144, 24576]
    acc_size = np.zeros((5,), 'uint32')
    for i in range(1, 5):
        acc_size[i] = size[i - 1] + acc_size[i - 1]
    flow5 = prediction[acc_size[0]: acc_size[1]].reshape(12, 16, 2)
    flow4 = prediction[acc_size[1]: acc_size[2]].reshape(24, 32, 2)
    flow3 = prediction[acc_size[2]: acc_size[3]].reshape(48, 64, 2)
    flow2 = prediction[acc_size[3]: acc_size[4]].reshape(96, 128, 2)
    return flow5, flow4, flow3, flow2
