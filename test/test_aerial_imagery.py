import cv2
import os
import matplotlib.pyplot as plt
import utils

ROOT = '/media/jack/data/Dataset/aerial imagery'
full = False

if full:
    left_image = cv2.imread(os.path.join(ROOT, '0-rectL.tif'))
    left_image = cv2.rotate(left_image, cv2.ROTATE_90_CLOCKWISE)
    left_image = utils.rgb2bgr(left_image)
    plt.imshow(left_image)

else:
    left_image = cv2.imread(os.path.join(ROOT, '0-rectL.tif'))
    right_image = cv2.imread(os.path.join(ROOT, '0-rectR.tif'))

    left_image = cv2.rotate(left_image, cv2.ROTATE_90_CLOCKWISE)
    right_image = cv2.rotate(right_image, cv2.ROTATE_90_CLOCKWISE)

    r = 2950
    c = 5760
    height = 800
    width = 1472
    disp = 400

    left_image = utils.rgb2bgr(left_image)
    right_image = utils.rgb2bgr(right_image)

    left_image = left_image[r:r + height, c:c + width]
    right_image = right_image[r:r + height, c - disp:c - disp + width]

    plt.subplot(121)
    plt.imshow(left_image)

    plt.subplot(122)
    plt.imshow(right_image)

plt.show()

