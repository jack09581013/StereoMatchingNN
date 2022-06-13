import numpy as np
import cv2
import sys
import numpngw

img = np.full((100, 100), 16.265087, dtype='float32')
img[:50, :50] = 0x7FFF

print(img[0, 0])  # 32770.0
print(img[99, 99])
print(img.astype('uint16')[99, 99])

# BGR
cv2.imwrite('image/png_test.png', (img*0x100).astype('uint16'))
img = cv2.imread('image/png_test.png')
print(img[99, 99])