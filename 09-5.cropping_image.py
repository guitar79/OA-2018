# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:15:44 2018

@author: user
"""
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
#%%
dir_name = '20170220_m35/'
f_name = 'g3035794_p.fits'

hdu = fits.open(dir_name+f_name)
img = hdu[0].data

plt.figure(figsize=(12,12))
ax = plt.gca()
im = plt.imshow(img, vmax=65536, origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

#%%
img = hdu[0].data[250:1200, 1000:1800]
#img = hdu[0].data
plt.figure(figsize=(12,12))
ax = plt.gca()
im = plt.imshow(img, vmax=65536, origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

#%%
from astropy.stats import mad_std, sigma_clipped_stats

FWHM   = 2.5
sky_th = 5    # sky_th * sky_sigma will be used for detection lower limit : sky_th = 5
sky_s  = mad_std(img)
thresh = sky_th*sky_s
print(' sky_s x sky_th = threshold')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s, sky_th, thresh))

# What if we do "sigma-clip" than MAD?
sky_a, sky_m, sky_s_sc = sigma_clipped_stats(img) # default is 3-sigma, 5 iters
thresh_sc = sky_th*sky_s_sc
print('3 sigma 5 iters clipped case:')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s_sc, sky_th, thresh_sc))

#%%
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from photutils import CircularAperture as CircAp

DAOfind = DAOStarFinder(fwhm=FWHM, threshold=thresh, 
                        sharplo=0.5, sharphi=2.0,  # default values: sharplo=0.2, sharphi=1.0,
                        roundlo=0.0, roundhi=0.2,  # default values: roundlo=-1.0, roundhi=1.0,
                        sigma_radius=1.5,          # default values: sigma_radius=1.5,
                        ratio=1.0,                 # 1.0: circular gaussian:  ratio=1.0,
                        exclude_border=False)       # To exclude sources near edges : exclude_border=True

# The DAOStarFinder object ("find") gets at least one input: the image.
# Then it returns the astropy table which contains the aperture photometry results:
DAOfound = DAOfind(img)
print('DAOfound \n', DAOfound)

if len(DAOfound)==0 :
    print ('No star founded using DAOStarFinder')
else : 
    # Use the object "found" for aperture photometry:
    # save XY coordinates:
    print (len(DAOfound), 'stars founded')
    DAOcoord = (DAOfound['xcentroid'], DAOfound['ycentroid']) 
    
    # Save apertures as circular, 4 pixel radius, at each (X, Y)
    DAOapert = CircAp(DAOcoord, r=4.)  
    print('DAOapert\n ', DAOapert)
    
    DAOimgXY = np.array(DAOcoord)
    print('DAOimgXY \n', DAOimgXY)

    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = plt.imshow(img, vmax=65536, origin='lower')
    DAOapert.plot(color='red', lw=2., alpha=0.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.show()
    
    #%%
import numpy as np
import matplotlib.pyplot as plt
from photutils import IRAFStarFinder
from photutils import CircularAperture as CircAp

IRAFfind = IRAFStarFinder(fwhm=FWHM, threshold=thresh,
                          sigma_radius=1.5, minsep_fwhm=6.5,  # default values: sigma_radius=1.5, minsep_fwhm=2.5,
                          sharplo=0.5, sharphi=2.0,   # default values: sharplo=0.5, sharphi=2.0,
                          roundlo=0.0, roundhi=0.2,   # default values: roundlo=0.0, roundhi=0.2,
                          sky=None, exclude_border=False)  # default values: sky=None, exclude_border=False)

# The DAOStarFinder object ("find") gets at least one input: the image.
# Then it returns the astropy table which contains the aperture photometry results:
IRAFfound = IRAFfind(img)
print('IRAFfound \n', IRAFfound)

if len(IRAFfound)==0 :
    print ('No star founded using IRAFStarFinder')
else : 
    # Use the object "found" for aperture photometry:
    # save XY coordinates:
    print (len(IRAFfound), 'stars founded')
    IRAFcoord = (IRAFfound['xcentroid'], IRAFfound['ycentroid']) 
    
    # Save apertures as circular, 4 pixel radius, at each (X, Y)
    IRAFapert = CircAp(IRAFcoord, r=4.)  
    #print('IRAFapert\n ', IRAFapert)
    
    IRAFimgXY = np.array(IRAFcoord)
    #print('IRAFimgXY \n', IRAFimgXY)

    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = plt.imshow(img, vmax=65536, origin='lower')
    IRAFapert.plot(color='red', lw=2., alpha=0.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.show()
    
#%%
'''
                    sharplo=0.2, sharphi=1.0,  # default values: sharplo=0.2, sharphi=1.0,
                    roundlo=-1.0, roundhi=1.0, # default values: roundlo=-1.0, roundhi=1.0,
                    sigma_radius=1.5,          # default values: sigma_radius=1.5,
                    ratio=1.0,                 # 1.0: circular gaussian:  ratio=1.0,
                    
                    apert Aperture: CircularAperture
                    positions: [[400.15607361, 317.93856704],
                    [180.20468795, 615.67729319]]
'''
