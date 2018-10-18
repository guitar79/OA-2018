# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome fits file
"""

from glob import glob
import numpy as np
import cv2
from astropy.io import fits
from astropy.stats import sigma_clip
import os

def fits_to_cv2(image_path):
    hdu = fits.open(image_path)
    data = hdu[0].data
    return np.array(data, dtype=np.uint16)

def align_image(im1, im2):
    # Convert images to grayscale
    #im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    #im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    im1_gray = im1
    im2_gray = im2
    im1_32f_gray = np.array(im1_gray/65536.0, dtype=np.float32)
    im2_32f_gray = np.array(im2_gray/65536.0, dtype=np.float32)
    
    # Find size of image1
    sz = im1.shape
     
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
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_32f_gray,im2_32f_gray,warp_matrix, warp_mode, criteria)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    # Show final results
    return im2_aligned

data_dir = '20181012.M42'
image_chls = ['L', 'R', 'G', 'B', 'H', 'S', 'O', 'U', 'B', 'V', 'R', 'I']
#image_chl = 'R'
for image_chl in image_chls : 
    img_list = sorted(glob(os.path.join(data_dir, '*'+image_chl+'.fit')))
    if len(img_list) > 2 :
        ref_image = img_list[0]
        images_to_align = img_list[0:]
        
        ref_image = fits_to_cv2(ref_image)
        aligned_images = []
        
        for image_to_align in images_to_align:
            image = fits_to_cv2(image_to_align)
            res_image = align_image(ref_image, image)
            aligned_images.append(res_image)
            #break
        print ('Succeed align images in channel', image_chl)
        #combine image using algned_images:
        mean_image = np.mean(aligned_images, axis=0).astype(dtype=np.uint16)
        median_image = np.median(aligned_images, axis=0).astype(dtype=np.uint16)
        
        sigma_clip_image = sigma_clip(aligned_images, sigma=3, \
                    sigma_lower=None, sigma_upper=None, iters=5, axis=None, copy=True)
        cv2.imwrite(data_dir+'/mean_image_'+image_chl+'.png', mean_image)
        cv2.imwrite(data_dir+'/median_image_'+image_chl+'.png', median_image)
        cv2.imwrite(data_dir+'/sigma_clip_image_'+image_chl+'.png', sigma_clip_image[0])
        print ('Succeed combine images in channel', image_chl)
    else : 
        print ('There is no images for align in channel', image_chl)

    


