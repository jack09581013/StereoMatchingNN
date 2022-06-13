import numpy as np
import cv2
import sys
import numpngw
import imageio

def read_png(file_path):
    with open(file_path, 'rb') as file:
        return bytearray(file.read())

def print_bytes(data: bytearray, bytes_per_line=40, amount=None):
    hex_string = data.hex().upper()
    count = 0
    if amount is None:
        amount = len(hex_string)
    elif amount > len(hex_string):
        amount = len(hex_string)//2

    for i in range(0, amount*2, 2):
        print(hex_string[i] + hex_string[i + 1] + ' ', end='')
        count += 2
        if count == bytes_per_line*2:
            print()
            count = 0

file_path = r'D:\Google Drive\Projects\Pycharm\StereoMatchingNN\result\prediction\last-version\GDFNet_mdc6f_benchmark\KITTI_2015_benchmark\disp_0\000000_10.png'
# file_path = r'D:\Dataset\KITTI 2015\training\disp_noc_0\000000_10.png'
img = cv2.imread(file_path)
# img = imageio.imread(file_path)
# np.set_printoptions(threshold=sys.maxsize)
print(img[250:280, 250:280].reshape(-1))
print(img.shape)

print(img.dtype)
print(img.reshape(-1).max())

# print(img)
# print_bytes(bytearray(img), amount=10)
# print()
# print_bytes(read_png(file_path), amount=50000)

# 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27
#  27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27
#  27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27
#  27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27
#  27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27
#  28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28
#  28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28
#  28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28
#  28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 29 29 29 29 29 29
#  29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29
#  29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29
#  29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29 29

