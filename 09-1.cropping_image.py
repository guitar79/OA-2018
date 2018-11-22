# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:15:44 2018

@author: user
"""

import matplotlib.pyplot as plt
from astropy.io import fits

hdu = fits.open('mean_image_R.fit')
img = hdu[0].data[300:1200, 300:1200]

plt.figure(figsize=(10,10))
plt.imshow(img, vmax=6500, origin='lower')
plt.colorbar()
plt.show()

from astropy.stats import mad_std, sigma_clipped_stats

FWHM   = 2.5
sky_th = 5    # sky_th * sky_sigma will be used for detection lower limit
sky_s  = mad_std(img)
thresh = sky_th*sky_s
print(' sky_s x sky_th = threshold')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s, sky_th, thresh))

# What if we do "sigma-clip" than MAD?
sky_a, sky_m, sky_s_sc = sigma_clipped_stats(img) # default is 3-sigma, 5 iters
thresh_sc = sky_th*sky_s_sc
print('3 sigma 5 iters clipped case:')
print('{0:8.6f} x {1:4d}   =   {2:8.6f}\n'.format(sky_s_sc, sky_th, thresh_sc))


import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from photutils import CircularAperture as CircAp

find   = DAOStarFinder(fwhm=FWHM, threshold=thresh,
                      sharplo=0.2, sharphi=1.0,  # default values 
                      roundlo=-1.0, roundhi=1.0, # default values
                      sigma_radius=1.5,          # default values
                      ratio=1.0,                 # 1.0: circular gaussian
                      exclude_border=True)       # To exclude sources near edges

# The DAOStarFinder object ("find") gets at least one input: the image.
# Then it returns the astropy table which contains the aperture photometry results:
found = find(img)

# Use the object "found" for aperture photometry:
# save XY coordinates:
coord = (found['xcentroid'], found['ycentroid']) 

# Save apertures as circular, 4 pixel radius, at each (X, Y)
apert = CircAp(coord, r=4.)  

# Draw image and overplot apertures:
plt.figure(figsize=(10,10))
plt.imshow(img, vmax=6550)
apert.plot(color='red', lw=2., alpha=0.7)
plt.colorbar()
plt.show()