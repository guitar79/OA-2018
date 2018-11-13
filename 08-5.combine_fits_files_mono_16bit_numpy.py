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
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt

img_names = ['NGC2244-001R_aligned.FIT', 'NGC2244-002R_aligned.FIT',\
            'NGC2244-003R_aligned.FIT','NGC2244-004R_aligned.FIT',\
            'NGC2244-005R_aligned.FIT','NGC2244-006R_aligned.FIT']

print('img_names',  img_names)
print('img_names[0]',  img_names[0])

#
list_images = []
for img_name in(img_names):
    hdu = fits.open(img_name)
    data = hdu[0].data
    image_data = np.array(data, dtype=np.uint16)
    list_images.append(image_data)
print('list_images',  list_images)
print('iist_images[0]',  list_images[0])
    
#combine image using algned_images:            
mean_image = np.mean(list_images, axis=0).astype(dtype=np.uint16)
cv2.imwrite('NGC2244-R-mean_image.png', mean_image)

# write 16bit monochrome fits file
hdu[0].data = mean_image
hdu.writeto('NGC2244-R-mean_image.fits', overwrite =True)

# show fits file 
plt.imshow(mean_image, cmap = 'gray', interpolation = 'None')
plt.show()
            
median_image = np.median(list_images, axis=0).astype(dtype=np.uint16)
cv2.imwrite('NGC2244-R-median_image.png', median_image)
hdu[0].data = median_image
hdu.writeto('NGC2244-R-median_image.fits', overwrite =True)

# show fits file 
plt.imshow(median_image, cmap = 'gray', interpolation = 'None')
plt.show()

sigma_clip_image = sigma_clip(list_images, sigma=3, \
            sigma_lower=None, sigma_upper=None, iters=5, axis=None, copy=True)
cv2.imwrite('NGC2244-R-sigma_clip_image.png', sigma_clip_image[0])
hdu[0].data = sigma_clip_image[0]
hdu.writeto('NGC2244-R-sigma_clip_image.fit', overwrite =True)

# show fits file 
plt.imshow(sigma_clip_image[0], cmap = 'gray', interpolation = 'None')
plt.show()
