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
sky_th = 20    # sky_th * sky_sigma will be used for detection lower limit : sky_th = 5
sky_s  = mad_std(img)
thresh = sky_th*sky_s
print('sky_s x sky_th = threshold')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s, sky_th, thresh))

# What if we do "sigma-clip" than MAD?
sky_a, sky_m, sky_s_sc = sigma_clipped_stats(img) # default is 3-sigma, 5 iters
thresh_sc = sky_th*sky_s_sc
print('3 sigma 5 iters clipped case:')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s_sc, sky_th, thresh_sc))

#%%
from photutils import detect_threshold

thresh = detect_threshold(data=img.data, snr=3)
thresh = thresh[0][0]
# This will give you 3*bkg_std.
print('detect_threshold', thresh)

#%%
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from photutils import CircularAperture as CircAp

DAOfind = DAOStarFinder(fwhm=FWHM, threshold=thresh, 
                        sharplo=0.5, sharphi=2.0,  # default values: sharplo=0.2, sharphi=1.0,
                        roundlo=0.0, roundhi=0.2,  # default values: roundlo=-1.0, roundhi=1.0,
                        sigma_radius=1.5,          # default values: sigma_radius=1.5,
                        ratio=0.5,                 # 1.0: circular gaussian:  ratio=1.0,
                        sky=None, exclude_border=False)       # To exclude sources near edges : exclude_border=True

# The DAOStarFinder object ("find") gets at least one input: the image.
# Then it returns the astropy table which contains the aperture photometry results:
DAOfound = DAOfind(img)

if len(DAOfound)==0 :
    print ('No star founded using DAOStarFinder\n'*3)
else : 
    # Use the object "found" for aperture photometry:
    # save XY coordinates:
    print (len(DAOfound), 'stars founded')
    #print('DAOfound \n', DAOfound)
    DAOfound.pprint(max_width=1800)
    DAOfound.write(dir_name+f_name[:-5]+'_DAOStarfinder.csv', overwrite=True, format='ascii.fast_csv')

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
                          sigma_radius=1.5, minsep_fwhm=1.0,  # default values: sigma_radius=1.5, minsep_fwhm=2.5,
                          sharplo=0.5, sharphi=2.0,   # default values: sharplo=0.5, sharphi=2.0,
                          roundlo=0.0, roundhi=0.2,   # default values: roundlo=0.0, roundhi=0.2,
                          sky=None, exclude_border=False)  # default values: sky=None, exclude_border=False)

# The DAOStarFinder object ("find") gets at least one input: the image.
# Then it returns the astropy table which contains the aperture photometry results:
IRAFfound = IRAFfind(img)

if len(IRAFfound)==0 :
    print ('No star founded using IRAFStarFinder')
else : 
    print('IRAFfound \n', IRAFfound)
    IRAFfound.pprint(max_width=1800)
    IRAFfound.write(dir_name+f_name[:-5]+'_IRAFStarfinder.csv', overwrite=True, format='ascii.fast_csv')
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

'''
The available formats are:
           Format           Read Write Auto-identify Deprecated
--------------------------- ---- ----- ------------- ----------
                      ascii  Yes   Yes            No           
               ascii.aastex  Yes   Yes            No           
                ascii.basic  Yes   Yes            No           
     ascii.commented_header  Yes   Yes            No           
                  ascii.csv  Yes   Yes            No           
                 ascii.ecsv  Yes   Yes           Yes           
           ascii.fast_basic  Yes   Yes            No           
ascii.fast_commented_header  Yes   Yes            No           
             ascii.fast_csv  Yes   Yes            No           
       ascii.fast_no_header  Yes   Yes            No           
             ascii.fast_rdb  Yes   Yes            No           
             ascii.fast_tab  Yes   Yes            No           
          ascii.fixed_width  Yes   Yes            No           
ascii.fixed_width_no_header  Yes   Yes            No           
 ascii.fixed_width_two_line  Yes   Yes            No           
                 ascii.html  Yes   Yes           Yes           
                 ascii.ipac  Yes   Yes            No           
                ascii.latex  Yes   Yes           Yes           
            ascii.no_header  Yes   Yes            No           
                  ascii.rdb  Yes   Yes           Yes           
                  ascii.rst  Yes   Yes            No           
                  ascii.tab  Yes   Yes            No           
                       fits  Yes   Yes           Yes           
                       hdf5  Yes   Yes           Yes           
                   jsviewer   No   Yes            No           
                    votable  Yes   Yes           Yes           
                     aastex  Yes   Yes            No        Yes
                        csv  Yes   Yes           Yes        Yes
                       html  Yes   Yes            No        Yes
                       ipac  Yes   Yes            No        Yes
                      latex  Yes   Yes            No        Yes
                        rdb  Yes   Yes            No        Yes
                        '''