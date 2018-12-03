# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:15:44 2018

@author: user
"""
#%%
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture as CircAp
from photutils import CircularAnnulus as CircAn
from photutils import aperture_photometry as APPHOT

#%%
#Returns magnitude from flux.
def mag_inst(flux, ferr):
    m_inst = -2.5 * np.log10(flux)
    merr   = 2.5/ np.log(10) * ferr / flux
    return m_inst, merr

def sky_fit(all_sky, method='mode', sky_nsigma=3, sky_iter=5, \
            mode_option='sex', med_factor=2.5, mean_factor=1.5):
    '''
    Estimate sky from given sky values.

    Parameters
    ----------
    all_sky : ~numpy.ndarray
        The sky values as numpy ndarray format. It MUST be 1-d for proper use.
    method : {"mean", "median", "mode"}, optional
        The method to estimate sky value. You can give options to "mode"
        case; see mode_option.
        "mode" is analogous to Mode Estimator Background of photutils.
    sky_nsigma : float, optinal
        The input parameter for sky sigma clipping.
    sky_iter : float, optinal
        The input parameter for sky sigma clipping.
    mode_option : {"sex", "IRAF", "MMM"}, optional.
        sex  == (med_factor, mean_factor) = (2.5, 1.5)
        IRAF == (med_factor, mean_factor) = (3, 2)
        MMM  == (med_factor, mean_factor) = (3, 2)

    Returns
    -------
    sky : float
        The estimated sky value within the all_sky data, after sigma clipping.
    std : float
        The sample standard deviation of sky value within the all_sky data,
        after sigma clipping.
    nsky : int
        The number of pixels which were used for sky estimation after the
        sigma clipping.
    nrej : int
        The number of pixels which are rejected after sigma clipping.
    -------

    '''
    sky = all_sky.copy()
    if method == 'mean':
        return np.mean(sky), np.std(sky, ddof=1)

    elif method == 'median':
        return np.median(sky), np.std(sky, ddof=1)

    elif method == 'mode':
        sky_clip   = sigma_clip(sky, sigma=sky_nsigma, iters=sky_iter)
        sky_clipped= sky[np.invert(sky_clip.mask)]
        nsky       = np.count_nonzero(sky_clipped)
        mean       = np.mean(sky_clipped)
        med        = np.median(sky_clipped)
        std        = np.std(sky_clipped, ddof=1)
        nrej       = len(all_sky) - len(sky_clipped)

        if nrej < 0:
            raise ValueError('nrej < 0: check the code')

        if nrej > nsky: # rejected > survived
            raise Warning('More than half of the pixels rejected.')

        if mode_option == 'IRAF':
            if (mean < med):
                sky = mean
            else:
                sky = 3 * med - 2 * mean

        elif mode_option == 'MMM':
            sky = 3 * med - 2 * mean

        elif mode_option == 'sex':
            if (mean - med) / std > 0.3:
                sky = med
            else:
                sky = (2.5 * med) - (1.5 * mean)
        else:
            raise ValueError('mode_option not understood')

        return sky, std, nsky, nrej

#%%
dir_name = '20170220_m35/'
file_list = sorted(glob(os.path.join(dir_name, '*_p.fit')))

for f_name in file_list[10:11]:
    print('f_name :', f_name)
    hdu = fits.open(f_name)
    img = np.array(hdu[0].data/65536.0, dtype=np.float32)
    
    # if image value < 10^(-6), replace the pixel as 10^(-6)
    img[img < 1.e-6] = 1.e-6
    
    FWHM   = 4.5
    sky_th = 10    # sky_th * sky_sigma will be used for detection lower limit : sky_th = 5
    
    # What if we do "sigma-clip" than MAD?
    #sigma_clipped_stats() will return the mean, median, and standard deviation of a sigma-clipped array:
    sky_a, sky_m, sky_s_sc = sigma_clipped_stats(img) # default is 3-sigma, 5 iters
    thresh = sky_th*sky_s_sc
    print('3 sigma 5 iters clipped case:')
    print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s_sc, sky_th, thresh))
    
    DAOfind = DAOStarFinder(fwhm=FWHM, threshold=thresh, 
                            sharplo=0.2, sharphi=3.0,  # default values: sharplo=0.2, sharphi=1.0,
                            roundlo=-1.0, roundhi=1.0,  # default values: roundlo=-1.0, roundhi=1.0,
                            sigma_radius=1.5,          # default values: sigma_radius=1.5,
                            ratio=0.9,                 # 1.0: circular gaussian:  ratio=1.0,
                            exclude_border=True)       # To exclude sources near edges : exclude_border=True
    
    # The DAOStarFinder object ("DAOfind") gets at least one input: the image.
    # Then it returns the astropy table which contains the aperture photometry results:
    DAOfound = DAOfind(img)
    
    if len(DAOfound)==0 :
        print ('No star was founded using DAOStarFinder\n'*3)
    else : 
        # Use the object "found" for aperture photometry:
        N_stars = len(DAOfound)
        print(N_stars, 'star(s) is(are) founded')
        #print('DAOfound \n', DAOfound)
        #DAOfound.pprint(max_width=1800)
        # save XY coordinates:
        DAOfound.write(f_name[:-4]+'_DAOStarFinder.csv', overwrite=True, format='ascii.fast_csv')
        DAOcoord = (DAOfound['xcentroid'], DAOfound['ycentroid']) 
        DAOannul = CircAn(positions=DAOcoord, r_in=4*FWHM, r_out=6*FWHM)
        
        # Save apertures as circular, 4 pixel radius, at each (X, Y)
        DAOapert = CircAp(DAOcoord, r=4.)  
        #print('DAOapert\n ', DAOapert)
        
        DAOimgXY = np.array(DAOcoord)
        #print('DAOimgXY \n', DAOimgXY)
        
        plt.figure(figsize=(16,12))
        ax = plt.gca()
        im = plt.imshow(img, vmax=thresh*4, origin='lower')
        DAOannul.plot(color='red', lw=2., alpha=0.7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(f_name[:-4]+'_DAOstarfinder_Annulus_result_all_stars.png', overwrite=True)
        #plt.show()
    
        #%%
        #print('Star ID    msky  sky_std  nsky nrej')
        #for star_ID in range(2, 10) : # for debug
        img_uint16 = np.array(img*65536.0, dtype=np.uint16)
        ronoise = hdu[0].header['RDNOISE']
        #gain =  0.46599999070167542     # e/ADU
        gain = hdu[0].header['GAIN']
        N_star = len(DAOfound)
        
        mag_ann  = np.zeros(N_star)
        merr_ann = np.zeros(N_star)
        
        # aperture sum
        apert_sum = APPHOT(img_uint16, DAOapert, method='exact')['aperture_sum']
        ap_area   = DAOapert.area()
        #print(apert_sum)
        
        apert_result = 'ID, Msky, sky_std, Sky count Pixel_N, Sky reject Pixel_N, mag_ann, merr_ann\n'
        
        for star_ID in range(0, N_stars)[10:12]:

            # since our `DAOannul` has many elements :
            mask_annul = (DAOannul.to_mask(method='center'))[star_ID]
            mask_apert = (DAOapert.to_mask(method='center'))[star_ID]
            # CAUTION!! YOU MUST USE 'center', NOT 'exact'!!!
    
            cutimg = mask_annul.cutout(img)
            #cutimg.tofile('{0!s}_DAOstarfinder_Star_Flux_pixel_value_starID_{1:04}.csv'.format(f_name[:-4], star_ID), sep=',')
            df_cutimg = pd.DataFrame(cutimg*65536.0, dtype=np.uint16)
            df_cutimg.to_csv('{0!s}_DAOstarfinder_Star_Flux_pixel_value_starID_{1:04}.csv'.format(f_name[:-4], star_ID))
            
            cut_apert = mask_apert.cutout(img)
            #cutimg.tofile('{0!s}_DAOstarfinder_Star_Flux_pixel_value_starID_{1:04}.csv'.format(f_name[:-4], star_ID), sep=',')
            df_cut_apert = pd.DataFrame(cut_apert*65536.0, dtype=np.uint16)
            df_cut_apert.to_csv('{0!s}_DAOstarfinder_Star_apertruer_Flux_pixel_value_starID_{1:04}.csv'.format(f_name[:-4], star_ID))
            
            sky_apply = mask_annul.multiply(img)  # change from 'sky_apply  = mask_annul.apply(img)'
            #sky_apply.tofile('{0!s}_DAOstarfinder_Sky_Annulus_pixel_value_starID_{1:04}.csv'.format(f_name[:-4], star_ID), sep=',')
            df_sky_apply = pd.DataFrame(sky_apply*65536.0, dtype=np.uint16)
            df_sky_apply.to_csv('{0!s}_DAOstarfinder_Sky_Annulus_pixel_value_starID_{1:04}.csv'.format(f_name[:-4], star_ID))
            
            sky_non0   = np.nonzero(sky_apply)
            sky_pixel  = sky_apply[sky_non0]
            msky, sky_std, nsky, nrej = sky_fit(sky_pixel, method='mode', mode_option='sex')
            
            # sky estimation
            sky_apply_uint16  = mask_annul.multiply(img_uint16)
            sky_non0_uint16   = np.nonzero(sky_apply_uint16)
            sky_pixel_uint16  = sky_apply[sky_non0_uint16]
            msky, sky_std, nsky, nrej = sky_fit(sky_pixel_uint16, method='mode', mode_option='sex')
               
            flux_star = apert_sum[star_ID] - msky * ap_area  # total - sky
            flux_err  = np.sqrt(apert_sum[star_ID] * gain    # Poissonian (star + sky)
                                + ap_area * ronoise**2 # Gaussian
                                + (ap_area * (gain * sky_std))**2 / nsky ) 
            mag_ann[star_ID], merr_ann[star_ID] = mag_inst(flux_star, flux_err)
            #print('{0:7d}: {1:.5f} {2:.5f} {3:4d} {4:3d} {5:.3f} {6:.3f}'.format(i, msky, sky_std, nsky, nrej, mag_ann[i], merr_ann[i]))
            apert_result += '{0}, {1:.5f}, {2:.5f}, {3:4d}, {4:3d}, {5:.3f}, {6:.3f}\n'.format(star_ID, msky, sky_std, nsky, nrej, mag_ann[star_ID], merr_ann[star_ID])
            
            fig = plt.figure(figsize=(12,5))
            fig.add_subplot(1,3,1)
            plt.imshow(cutimg, vmin=0.1, vmax=thresh*4, origin='lower')
            plt.xlabel('meam: '+str(np.mean(cutimg))+'\n'+'max: '+str(np.mean(cutimg)))
            fig.add_subplot(1,3,2)
            plt.imshow(cut_apert, vmin=0.1, vmax=thresh*4, origin='lower')
            plt.xlabel('sum: '+str(np.sum(cut_apert))+'\n mean: '+str(np.mean(cut_apert))+'\n max: '+str(np.mean(cut_apert))+'\n Number of Pixel :'+str(len(cut_apert)))
            fig.add_subplot(1,3,3)
            plt.imshow(sky_apply, vmin=0.1, vmax=thresh*4, origin='lower')
            plt.xlabel('sum: '+str(np.sum(sky_apply))+'\n mean: '+str(np.mean(sky_apply))+'\n max: '+str(np.mean(sky_apply))+'\n Number of Pixel :'+str(len(sky_apply)))
            plt.gcf().suptitle('DAOstarfinder Annulus result \n filename : {0!s} \n (star ID) : {1:04}'.format(f_name, star_ID), fontsize=12)
            plt.savefig('{0!s}_DAOstarfinder_Annulus_result_starID_{1:04}.png'.format(f_name[:-4], star_ID), overwrite=True)
            #plt.show()
        print(apert_result)
        with open(f_name[:-4]+'_AP_Annulus_result_all.csv', 'w') as f:
            f.write(apert_result)
                
        #%%
        plt.figure(figsize=(16,12))
        plt.errorbar([i], msky, yerr=sky_std, capsize=3, marker='o', color='b')
        plt.xlabel('Star ID', fontsize=16)
        plt.ylabel('msky +- sky_std', fontsize=16)
        plt.xticks(np.arange(0, N_stars, step=int(N_stars/10)))
        plt.grid(ls=':')
        plt.savefig(f_name[:-4]+'_AP_Annulus_Msky_sky_std_result.png', overwrite=True)
        #plt.show()
           
        plt.figure(figsize=(16,12))
        plt.errorbar(i, mag_ann[i], yerr=merr_ann[i], marker='x', ms=10, capsize=3)
        plt.xlabel('Star ID', fontsize=16)
        plt.ylabel('Instrumental mag', fontsize=16)
        plt.xticks(np.arange(0, N_stars, step=int(N_stars/10)))
        plt.grid(ls=':')
        plt.savefig(f_name[:-4]+'_AP_Annulus_result.png', overwrite=True)
        #plt.show()
        
        #print and save result
        
                    
        