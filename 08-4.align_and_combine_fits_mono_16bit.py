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

img_name1 = 'NGC2244-001H.fit'
img_name2 = 'NGC2244-002H.fit'

#open fits file and make numpy array...
hdu1 = fits.open(img_name1)
data1 = hdu1[0].data
image1 = np.array(data1, dtype=np.uint16)
hdu2 = fits.open(img_name2)
data2 = hdu2[0].data
image2 = np.array(data2, dtype=np.uint16)
print('image1 data(uint16)\n', image1)
print('image2 data(uint16)\n', image2)

#align 2 images
#code from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
#convert int16 to float32
im1 = np.array(image1/65536.0, dtype=np.float32)
im2 = np.array(image2/65536.0, dtype=np.float32)
print('image1 data(float32)\n', im1)
print('image1 data(float32)\n', im2)

# Find size of image1
sz = im1.shape
print('image1 size :', sz)

# Define the motion model
warp_mode = cv2.MOTION_TRANSLATION
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
# Specify the number of iterations.
number_of_iterations = 5000;

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1, im2, warp_matrix, warp_mode, criteria)
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography 
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
       
print ('Succeed in alignment the', img_name1, 'and ', img_name2, 'files ......')
print ('im2_aligned(float32)\n', im2_aligned )

# convert to uint16
im2_aligned = np.uint16(im2_aligned*65536.0)
print ('im2_aligned(uint16)\n', im2_aligned)

# Save final results to fits file
hdu2[0].data = im2_aligned
hdu2.writeto(img_name2[:-4]+'_aligned'+img_name2[-4:], overwrite =True)
fits.setval(img_name2[:-4]+'_aligned'+img_name2[-4:], 'NOTES', value='aligned by guitar79@naver.com')
print (img_name2[:-4]+'_aligned'+img_name2[-4:], 'is saved ...')

# show fits file 
plt.imshow(im2_aligned, cmap = 'gray', interpolation = 'None')
plt.show()