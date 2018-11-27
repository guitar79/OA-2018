# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:15:44 2018

@author: user
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
dir_name = '20170220_m35/'
f_name = 'g3035794_p.fits'

hdu = fits.open(dir_name+f_name)
img = np.array(hdu[0].data[300:800, 1100:1700]/65536.0, dtype=np.float32)

# if image value < 10^(-6), replace the pixel as 10^(-6)
img[img < 1.e-6] = 1.e-6

plt.figure(figsize=(12,12))
ax = plt.gca()
im = plt.imshow(img, vmax=0.35, origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

#%%
from astropy.stats import mad_std, sigma_clipped_stats

FWHM   = 2.7
sky_th = 100    # sky_th * sky_sigma will be used for detection lower limit : sky_th = 5

sky_s  = mad_std(img)
thresh = sky_th*sky_s
print('sky_s x sky_th = threshold')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s, sky_th, thresh))

# What if we do "sigma-clip" than MAD?
sky_a, sky_m, sky_s_sc = sigma_clipped_stats(img) # default is 3-sigma, 5 iters
thresh_sc = sky_th*sky_s_sc
thresh = sky_th*sky_s_sc
print('3 sigma 5 iters clipped case:')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s_sc, sky_th, thresh_sc))

#%%
from photutils import detect_threshold

thresh_snr = detect_threshold(data=img.data, snr=3)
thresh_snr = thresh_snr[0][0]
# This will give you 3*bkg_std.
print('detect_threshold', thresh_snr)

#%%
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from photutils import CircularAperture as CircAp
from photutils import CircularAnnulus as CircAn

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
    print (len(DAOfound), 'stars were founded')
    #print('DAOfound \n', DAOfound)
    DAOfound.pprint(max_width=1800)
    # save XY coordinates:
    DAOfound.write(dir_name+f_name[:-5]+'_DAOStarFinder.csv', overwrite=True, format='ascii.fast_csv')
    DAOcoord = (DAOfound['xcentroid'], DAOfound['ycentroid']) 
    DAOannul = CircAn(positions=DAOcoord, r_in=4*FWHM, r_out=6*FWHM)
    
    # Save apertures as circular, 4 pixel radius, at each (X, Y)
    DAOapert = CircAp(DAOcoord, r=4.)  
    #print('DAOapert\n ', DAOapert)
    
    DAOimgXY = np.array(DAOcoord)
    #print('DAOimgXY \n', DAOimgXY)
    
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = plt.imshow(img, vmax=0.35, origin='lower')
    DAOannul.plot(color='red', lw=2., alpha=0.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

#%%
from photutils import IRAFStarFinder
from photutils import CircularAnnulus as CircAn
from photutils import CircularAperture as CircAp

IRAFfind = IRAFStarFinder(fwhm=FWHM, threshold=thresh,
                          sigma_radius=1.5, minsep_fwhm=2.5,  # default values: sigma_radius=1.5, minsep_fwhm=2.5,
                          sharplo=0.2, sharphi=3.0,   # default values: sharplo=0.5, sharphi=2.0,
                          roundlo=0.0, roundhi=0.5,   # default values: roundlo=0.0, roundhi=0.2,
                          sky=None, exclude_border=True)  # default values: sky=None, exclude_border=False)

# The DAOStarFinder object ("find") gets at least one input: the image.
# Then it returns the astropy table which contains the aperture photometry results:
IRAFfound = IRAFfind(img)

if len(IRAFfound)==0 :
    print ('No star founded using IRAFStarFinder')
else : 
    # Use the object "found" for aperture photometry:
    print (len(IRAFfound), 'stars founded')
    #print('IRAFfound \n', IRAFfound)
    IRAFfound.pprint(max_width=1800)
    # save XY coordinates:
    IRAFfound.write(dir_name+f_name[:-5]+'_IRAFStarFinder.csv', overwrite=True, format='ascii.fast_csv')
    IRAFcoord = (IRAFfound['xcentroid'], IRAFfound['ycentroid']) 
    IRAFannul = CircAn(positions=IRAFcoord, r_in=4*FWHM, r_out=6*FWHM)
    
    # Save apertures as circular, 4 pixel radius, at each (X, Y)
    IRAFapert = CircAp(IRAFcoord, r=4.)  
    #print('IRAFapert\n ', IRAFapert)
    
    IRAFimgXY = np.array(IRAFcoord)
    #print('IRAFimgXY \n', IRAFimgXY)
            
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = plt.imshow(img, vmax=0.35, origin='lower')
    IRAFannul.plot(color='red', lw=2., alpha=0.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

#%%
# since our `annul` has many elements, let me use [3] to use only the 4th annulus:
mask_annul = (DAOannul.to_mask(method='center'))[3]
# CAUTION!! YOU MUST USE 'center', NOT 'exact'!!!

cutimg = mask_annul.cutout(img)
plt.imshow(cutimg, vmin=0.1, vmax=0.35, origin='lower')
plt.show()

#%%
sky_apply  = mask_annul.multiply(img)  # change from 'sky_apply  = mask_annul.apply(img)'
plt.imshow(sky_apply, origin='lower', vmin=0.1, vmax=0.35)
plt.show()

#%%
import numpy as np
from astropy.stats import sigma_clip



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
sky_apply  = mask_annul.multiply(img)
sky_non0   = np.nonzero(sky_apply)
sky_pixel  = sky_apply[sky_non0]
msky, sky_std, nsky, nrej = sky_fit(sky_pixel, method='mode', mode_option='sex')
print(msky, sky_std, nsky, nrej)

#%%
N_stars = len(DAOfound)
print('Star ID    msky  sky_std     nsky nrej')
for i in range(0, N_stars):
    mask_annul = (DAOannul.to_mask(method='center'))[i]
    sky_apply  = mask_annul.multiply(img)
    sky_non0   = np.nonzero(sky_apply)
    sky_pixel  = sky_apply[sky_non0]
    msky, sky_std, nsky, nrej = sky_fit(sky_pixel, method='mode', mode_option='sex')
    print('{0:7d}: {1:.5f} {2:.5f} {3:4d} {4:3d}'.format(i, msky, sky_std, nsky, nrej))
    plt.errorbar([i], msky, yerr=sky_std, capsize=3, color='b')

plt.xlabel('Star ID')
plt.ylabel('msky +- sky_std')
plt.grid(ls=':')
plt.show()