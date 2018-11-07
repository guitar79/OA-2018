# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only caonn cr2 file

ModuleNotFoundError: No module named 'rawpy'
pip install rawpy

"""


from glob import glob
import numpy as np
import cv2
from astropy.io import fits
from astropy.stats import sigma_clip
import os

import rawpy
import imageio

image_path = '20120123.Taurus.5dm2/IMG_8366.CR2'
raw = rawpy.imread(image_path)
rgb = raw.postprocess(output_bps=16)
    
data_dir = '20120123.Taurus.5dm2'
#img_chls = ['L', 'R', 'G', 'B', 'H', 'S', 'O', 'u', 'b', 'v', 'r', 'i']
#img_chls = ['L', 'R', 'G', 'B']
img_chls = ['L', '']

#If you wnat to change the refrence file, please input the file order number...
ref_image_no = 0

#variable for saving files (True or False)
fit_save = False
png_save = True
tif_save = True

# read 16bit monochrome fits file and make numpy array
def cr2_to_cv2(image_path):
    raw = rawpy.imread(image_path)
    rgb = raw.postprocess(output_bps=16)
    return np.array(rgb, dtype=np.uint16)

def fits_to_cv2(image_path):
    hdu = fits.open(image_path)
    data = hdu[0].data
    return np.array(data, dtype=np.uint16)

def save_fits_after_align(image_path, res_data):
    hdu = fits.open(image_path)
    hdu[0].data = res_data
    hdu.writeto(image_path[:-4]+'_aligned'+image_path[-4:], overwrite =True)
    if os.path.isfile(image_path[:-4]+'_aligned'+image_path[-4:]) :
        fits.setval(image_path[:-4]+'_aligned'+image_path[-4:], 'NOTES', value='aligned by guitar79@naver.com')
        return print('aligned fits file is created')
    else : 
        return print('failure creating aligned fits file...')

#code from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
def align_image(im1, im2):
    #change to 16 bit mono
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    #im1_gray = im1
    #im2_gray = im2
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
    (cc, warp_matrix) = cv2.findTransformECC (im1_32f_gray, im2_32f_gray, warp_matrix, warp_mode, criteria)
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    # Show final results
    return im2_aligned

img_lists = sorted(glob(os.path.join(data_dir, '*.cr2')))
img_lists = [element.upper() for element in img_lists] ; img_lists

#for img_list in sorted(glob(os.path.join(data_dir, '*.fit'))) : 
ref_image = cr2_to_cv2(img_lists[ref_image_no])
print('reference image is', img_lists[ref_image_no])

if len(img_lists) > 2 :
    
    for img_chl in img_chls :
        imgs_to_align = list(filter(lambda x: img_chl+'.CR2' in x, img_lists))
        print(imgs_to_align)
        #imgs_to_align = [img_list for img_list in enumerate(img_lists) if img_list[-5] == img_chl+'.FIT']

        try : 
            print ('*'*60)
            print ('Starting alignment images in channel', img_chl)
            aligned_images = []
            for img_to_align in imgs_to_align:
                img = cr2_to_cv2(img_to_align)
                res_image = align_image(ref_image, img)
                aligned_images.append(res_image)
                print ('Succeed in alignment the', img_to_align, 'file ......')
                if png_save == True:
                    imageio.imsave(img_to_align[:-4]+'_aligned'+img_chl+'.png', res_image)
                    print('aligned png file is created')
                    #save_fits_after_align(img_to_align, res_image)
                if tif_save == True:
                    imageio.imsave(img_to_align[:-4]+'_aligned'+img_chl+'.tif', res_image)
                    print('aligned tif file is created')
                    #save_fits_after_align(img_to_align, res_image)
                if fit_save == True :
                    save_fits_after_align(img_to_align, res_image)
                    print('aligned fits file is created')
                    
            print(len(aligned_images), 'images in channel', img_chl, 'are aligned')
            
            #combine image using algned_images
            mean_image = np.mean(aligned_images, axis=0).astype(dtype=np.uint16)
            imageio.imsave(data_dir+'/mean_image_'+img_chl+'.tif', mean_image)
            #cv2.imwrite(data_dir+'/mean_image_'+img_chl+'.png', mean_image)
            print ('Succeed in combining', len(imgs_to_align), 'images on channel', img_chl, '(average)')
            
            median_image = np.median(aligned_images, axis=0).astype(dtype=np.uint16)
            cv2.imwrite(data_dir+'/median_image_'+img_chl+'.tif', median_image)
            print ('Succeed in combining', len(imgs_to_align), 'images on channel', img_chl, '(median)')
            
            sigma_clip_image = sigma_clip(aligned_images, sigma=3, \
                        sigma_lower=None, sigma_upper=None, iters=5, axis=None, copy=True)
            cv2.imwrite(data_dir+'/sigma_clip_image_'+img_chl+'.png', sigma_clip_image[0])
            print ('Succeed in combining', len(imgs_to_align), 'images on channel', img_chl, '(sigma clip)')
            
        except Exception as err: 
            print ('Error messgae .......')
            print (err)

else : 
    print ('There is no images for alignment')