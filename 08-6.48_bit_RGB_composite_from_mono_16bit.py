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
hdu_R = fits.open('mean_image_R.fit')
data_R = hdu_R[0].data
image_R = np.array(data_R/65536.0, dtype=np.float32)
hdu_G = fits.open('mean_image_G.fit')
data_G = hdu_G[0].data
image_G = np.array(data_G/65536.0, dtype=np.float32)
hdu_B = fits.open('mean_image_B.fit')
data_B = hdu_B[0].data
image_B = np.array(data_B/65536.0, dtype=np.float32)

print('image_R :', image_R)
print('image_G :', image_G)
print('image_B :', image_B)

# make empty array for RGB image
RGB = np.zeros((image_R.shape[0], image_R.shape[1], 3), dtype=np.float32)
# insert each channel image
RGB[:,:,0] = image_B
RGB[:,:,1] = image_G
RGB[:,:,2] = image_R
print('RGB :', RGB)

# display image
plt.imshow(RGB, interpolation = 'None', aspect='equal')
plt.show()

# convert 16bit each channel
RGB = np.asarray(RGB*65536.0, dtype=np.uint16)
print('RGB :', RGB)

# write 48 bit RGB png file
cv2.imwrite('RGB-48bit.png', RGB)

# write 48bit RGB fits file
hdu_R[0].data = RGB
hdu_R.writeto('RGB-bit.fit', overwrite =True)
