# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:15:44 2018

@author: user
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
#%%
dir_name = '20170220_m35/'
f_name = 'g3035794_p.fit'

hdu = fits.open(dir_name+f_name)
img = np.array(hdu[0].data/65536, dtype=np.float32)

plt.figure(figsize=(12,12))
ax = plt.gca()
im = plt.imshow(img, vmax=0.35, origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

#%%
img = np.array(hdu[0].data[200:1000, 700:1800]/65536.0, dtype=np.float32)
#img = hdu[0].data
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
sky_th = 10    # sky_th * sky_sigma will be used for detection lower limit : sky_th = 5
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

DAOfind = DAOStarFinder(fwhm=FWHM, threshold=thresh_snr, 
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
    
    # Save apertures as circular, 4 pixel radius, at each (X, Y)
    DAOapert = CircAp(DAOcoord, r=4.)  
    #print('DAOapert\n ', DAOapert)
    
    DAOimgXY = np.array(DAOcoord)
    #print('DAOimgXY \n', DAOimgXY)
    
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = plt.imshow(img, vmax=0.35, origin='lower')
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

IRAFfind = IRAFStarFinder(fwhm=FWHM, threshold=thresh_snr,
                          sigma_radius=1.5, minsep_fwhm=2.5,  # default values: sigma_radius=1.5, minsep_fwhm=2.5,
                          sharplo=0.2, sharphi=3.0,   # default values: sharplo=0.5, sharphi=2.0,
                          roundlo=0.0, roundhi=0.5,   # default values: roundlo=0.0, roundhi=0.2,
                          sky=None, exclude_border=True)  # default values: sky=None, exclude_border=False)

# The IRAFStarFinder object ("IRAFfind") gets at least one input: the image.
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
    
    # Save apertures as circular, 4 pixel radius, at each (X, Y)
    IRAFapert = CircAp(IRAFcoord, r=4.)  
    #print('IRAFapert\n ', IRAFapert)
    
    IRAFimgXY = np.array(IRAFcoord)
    #print('IRAFimgXY \n', IRAFimgXY)
            
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = plt.imshow(img, vmax=0.35, origin='lower')
    IRAFapert.plot(color='red', lw=2., alpha=0.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


