# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome fits file

ModuleNotFoundError: No module named 'cv2' 
conda install opencv
"""

import numpy as np
import cv2
from astropy.io import fits
import matplotlib.pyplot as plt

# read 16bit monochrome fits file
hdu_R = fits.open('median_image_R.fit')
image_R = np.array(hdu_R[0].data, dtype=np.uint16)
hdu_G = fits.open('median_image_G.fit')
image_G = np.array(hdu_G[0].data, dtype=np.uint16)
hdu_B = fits.open('median_image_B.fit')
image_B = np.array(hdu_B[0].data, dtype=np.uint16)

print('image_R :', image_R)
print('image_G :', image_G)
print('image_B :', image_B)

# make empty array for openCV RGB image
RGB = np.zeros((image_R.shape[0], image_R.shape[1], 3), dtype=np.uint16)
print('RGB.shape :', RGB.shape)
# insert each channel image for openCV RGB image
RGB[:,:,0] = image_B
RGB[:,:,1] = image_G
RGB[:,:,2] = image_R
print('RGB :', RGB)
print('RGB.shape :', RGB.shape)

# write 48 bit RGB png file
cv2.imwrite('RGB-48bit.png', RGB)

# insert each channel image for fits image
RGB1 = np.zeros((3, image_R.shape[0], image_R.shape[1]), dtype=np.uint16)
RGB1[0,:,:] = RGB[:,:,2]
RGB1[1,:,:] = RGB[:,:,1]
RGB1[2,:,:] = RGB[:,:,0]

# convert 16bit each channel
print('RGB :', RGB1)

# create 48bit fits file
hdu = fits.PrimaryHDU(RGB1)
hdul = fits.HDUList([hdu])
hdul[0].data
hdul[0].data.shape

hdul.writeto('RGB-48bit1.fit', overwrite =True)

print ('hdul.info()', hdul.info())
print ('hdul[0].header', hdul[0].header)

