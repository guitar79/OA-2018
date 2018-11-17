# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome fits file

ModuleNotFoundError: No module named 'cv2' 

conda install opencv
"""

from glob import glob
import numpy as np
import cv2
from astropy.io import fits
from astropy.stats import sigma_clip
import os

dir_name = '20151211.IC434/'
#img_chls = ['L', 'R', 'G', 'B', 'H', 'S', 'O', 'u', 'b', 'v', 'r', 'i']
img_chls = ['L', 'R', 'G', 'B', 'H']

#If you wnat to change the refrence file, please input the file order number...
ref_image_no = 0

#variable for saving fits file (True or False)
fits_save = False
png_save = True
tif_save = False
composite_process = True

# read 16bit monochrome fits file and make numpy array
def fits_to_cv2_uint16(image_path):
    hdu = fits.open(image_path)
    data = hdu[0].data
    return np.array(data, dtype=np.uint16)

#https://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python
def find_outlier_pixels(data, tolerance=1.0, worry_about_edges=True):
    #This function finds the hot or dead pixels in a 2D dataset. 
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)  #size=2
    difference = np.array(data, dtype=np.int32) - np.array(blurred, dtype=np.int32)
    difference = np.array(abs(difference), dtype=np.uint16)

    threshold = tolerance*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]

    if worry_about_edges == True:
        height,width = np.shape(data)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###
        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med

    return hot_pixels,fixed_image


#code from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
def align_image(im1, im2):
    #change to 16 bit mono
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
    (cc, warp_matrix) = cv2.findTransformECC (im1_32f_gray, im2_32f_gray, warp_matrix, warp_mode, criteria)
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    # Show final results
    return im2_aligned

def save_fits_after_align(image_path, res_data):
    hdu = fits.open(image_path)
    hdu[0].data = res_data
    hdu.writeto(image_path[:-4]+'_aligned'+image_path[-4:], overwrite =True)
    if os.path.isfile(image_path[:-4]+'_aligned'+image_path[-4:]) :
        fits.setval(image_path[:-4]+'_aligned'+image_path[-4:], 'NOTES', value='aligned by guitar79@naver.com')
        return print('aligned fits file is created')
    else : 
        return print('failure creating aligned fits file...')
    


### Start process
img_lists = sorted(glob(os.path.join(dir_name, '*.fit')))

#change to uppercate
img_lists = [element.upper() for element in img_lists] ; img_lists

#for img_list in sorted(glob(os.path.join(data_dir, '*.fit'))) : 
ref_image = fits_to_cv2_uint16(img_lists[ref_image_no])
ref_hdu = fits.open(img_lists[ref_image_no])
print('reference image is', img_lists[ref_image_no])

if len(img_lists) > 2 :
    
    for img_chl in img_chls :
        imgs_to_process = list(filter(lambda x: img_chl+'.FIT' in x, img_lists))
        print(imgs_to_process)
        #imgs_to_align = [img_list for img_list in enumerate(img_lists) if img_list[-5] == img_chl+'.FIT']

        try : 
            print ('*'*60)
            print ('Starting alignment images in channel', img_chl)
            aligned_images = []
            for img_to_process in imgs_to_process :
                img_cv2 = fits_to_cv2_uint16(img_to_process)
                hot_pixels, fixed_cv2 = find_outlier_pixels(img_cv2)
                #hot_pixels, fixed_cv2 = find_outlier_pixels(fixed_cv2)
                res_image = align_image(ref_image, fixed_cv2)
                aligned_images.append(res_image)
                print ('Succeed in alignment the', img_to_process, 'file ......')
                if fits_save == True:
                    save_fits_after_align(img_to_process, res_image)
                #break
            print(len(aligned_images), 'images in channel', img_chl, 'are aligned')
            #combine image using algned_images:            
            mean_image = np.mean(aligned_images, axis=0).astype(dtype=np.uint16)
            
            hot_pixels, mean_image = find_outlier_pixels(mean_image)
            print ('Succeed in combining', len(imgs_to_process), 'images on channel', img_chl, '(average)')
            if png_save == True :
                cv2.imwrite(dir_name+img_chl+'_mean_image.png', mean_image)
            
            ref_hdu[0].data = mean_image
            ref_hdu.writeto(dir_name+img_chl+'_mean_image.fit', overwrite =True)
                        
            median_image = np.median(aligned_images, axis=0).astype(dtype=np.uint16)
            
            hot_pixels, median_image = find_outlier_pixels(median_image)
            print ('Succeed in combining', len(imgs_to_process), 'images on channel', img_chl, '(median)')
            if png_save == True :
                cv2.imwrite(dir_name+img_chl+'_median_image.png', median_image)
            ref_hdu[0].data = median_image
            ref_hdu.writeto(dir_name+img_chl+'_median_image.fit', overwrite =True)
            
            sigma_clip_image = sigma_clip(aligned_images, sigma=3, \
                        sigma_lower=None, sigma_upper=None, iters=5, axis=None, copy=True)
            
            hot_pixels, sigma_clip_image[0] = find_outlier_pixels(sigma_clip_image[0])
            print ('Succeed in combining', len(imgs_to_process), 'images on channel', img_chl, '(sigma clip)')
            if png_save == True :
                cv2.imwrite(dir_name+img_chl+'_sigma_clip_image.png', sigma_clip_image[0])
            ref_hdu[0].data = sigma_clip_image[0]
            ref_hdu.writeto(dir_name+img_chl+'_sigma_clip_image.fit', overwrite =True)
            
        except Exception as err: 
            print ('Error messgae .......')
            print (err)

else : 
    print ('There is no images for alignment')

if composite_process == True :
    result_stacks = ['mean', 'median', 'sigma_clip']
    for result_stack in result_stacks :
        # read 16bit monochrome fits file
        hdu_R = fits.open(dir_name+'R_'+result_stack+'_image.fit')
        image_R = np.array(hdu_R[0].data/65536.0, dtype=np.float32)
        hdu_G = fits.open(dir_name+'G_'+result_stack+'_image.fit')
        image_G = np.array(hdu_G[0].data/65536.0, dtype=np.float32)
        hdu_B = fits.open(dir_name+'B_'+result_stack+'_image.fit')
        image_B = np.array(hdu_B[0].data/65536.0, dtype=np.float32)
    
        # make empty array for RGB image
        RGB = np.zeros((image_R.shape[0], image_R.shape[1], 3), dtype=np.float32)
        # insert each channel image
        RGB[:,:,0] = image_B
        RGB[:,:,1] = image_G
        RGB[:,:,2] = image_R
        
        # write 48 bit RGB png file
        cv2.imwrite(dir_name+result_stack+'_RGB-48bit.png', RGB)
        print(dir_name+result_stack+'RGB-48bit.png is created...')

        # insert each channel image for fits image
        RGB1 = np.zeros((3, image_R.shape[0], image_R.shape[1]), dtype=np.uint16)
        RGB1[0,:,:] = RGB[:,:,2]
        RGB1[1,:,:] = RGB[:,:,1]
        RGB1[2,:,:] = RGB[:,:,0]

        hdu = fits.PrimaryHDU(RGB1)
        hdul = fits.HDUList([hdu])

        hdul.writeto(dir_name+result_stack+'_RGB-48bit.fit', overwrite = True)
        print(dir_name+result_stack+'RGB-48bit.fit is created...')
    
        # write 48bit RGB fits file
        hdu_R[0].data = RGB
        hdu_R.writeto(dir_name+result_stack+'_RGB-48bit.fit', overwrite = True)
        print(dir_name+result_stack+'RGB-48bit.fit is created...')
        
else : 
    print ('48bit RGB file is not created. \n If you wnat make 48bit RGB file, set composite_process = True.... ')

