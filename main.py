#Script that uses opencv to detect objects in an image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Load the image
img = np.load('simpleworld_1/camera_data_Cam1simpleworld_1.npy')
img_unbuffered = np.frombuffer(img, np.uint8)
#print(img_unbuffered.shape)
img_cam1 = np.reshape(img_unbuffered, (1080, 1920, 3))
# print(img.shape)
cv2.imshow('image', img_cam1)
cv2.waitKey(0)

img = np.load('simpleworld_1/camera_data_Cam2simpleworld_1.npy')
img_unbuffered = np.frombuffer(img, np.uint8)
#print(img_unbuffered.shape)
img_cam2 = np.reshape(img_unbuffered, (1080, 1920, 3))
# print(img.shape)
cv2.imshow('image', img_cam2)
cv2.waitKey(0)

img = np.load('simpleworld_1/camera_data_Cam3simpleworld_1.npy')
img_unbuffered = np.frombuffer(img, np.uint8)
#print(img_unbuffered.shape)
img_cam3 = np.reshape(img_unbuffered, (1080, 1920, 3))
# print(img.shape)
cv2.imshow('image', img_cam3)
cv2.waitKey(0)

def mask_function(img): 
    color_dict = {'red': [0, 0, 255.0], 'blue': [255.0,0,0], 'yellow': [0,255.0,255.0], 'grey': [127.5,127.5,127.5],
    'pink': [255.0,0,255.0], 'orange': [0,127.5, 255.0], 'white': [255.0,255.0,255.0], 'brown': [0,15.3,63.75]}
    threshold = 40.0
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
        cv2.imshow(color, result_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

mask_function(img_cam1)
mask_function(img_cam2)
mask_function(img_cam3)


