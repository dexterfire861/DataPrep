#Script that uses opencv to detect objects in an image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Load the image
img = np.load('camera_data_Cam1simpleworld_1.npy')
img_unbuffered = np.frombuffer(img, np.uint8)
#print(img_unbuffered.shape)
img = np.reshape(img_unbuffered, (1080, 1920, 3))
# print(img.shape)
cv2.imshow('image', img)
cv2.waitKey(0)
# blue_mask = (img[:,:,2] == 0) & (img[:,:,1] == 0) & (img[:,:,0] == 255)

# blue_pixels = img.copy()
# blue_pixels[~blue_mask] = 0
# blue_image = Image.fromarray(blue_pixels)
# blue_image.show()

# red_mask = (img[:,:,2] == 255) & (img[:,:,1] == 0) & (img[:,:,0] == 0)

# red_pixels = img.copy()
# red_pixels[~red_mask] = 0
# red_image = Image.fromarray(red_pixels)
# red_image.show()
color_dict = {'red': [255.0, 0, 0], 'blue':[0,0,255.0], }
threshold = 10.0
for color in color_dict:
    target_rgb = color_dict[color]
    diffs = img[:, :] - target_rgb
    print(diffs.shape)
    euclid = diffs[:, :, 0] **2 + diffs[:, :, 1]**2
    print(euclid.shape)
    euclid += diffs[:, :, 2]**2
    euclid = np.sqrt(euclid)
    result_mask = np.where(euclid < threshold, 1.0, 0.0)
    result_mask = result_mask.astype(np.float32)
    cv2.imshow('image', result_mask)
    cv2.waitKey(0)