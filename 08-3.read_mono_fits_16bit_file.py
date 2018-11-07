# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome fits file

"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

#dir_name = '20181012.M42'
f_name = 'combined.fits'
fileinfo = f_name.split('.')

# read 16bit monochrome fits file
hdu = fits.open(f_name)
print('hdu :', hdu)
print('dir(hdu)\n', dir(hdu))

print('hdu.info()\n', hdu.info())
print('hdu[0].header\n', hdu[0].header)
print('hdu[0].data\n', hdu[0].data)
hdu[0].data = np.array(hdu[0].data/4.0, dtype=np.uint16)

# save data from fits file
fits_data = hdu[0].data
print('fits_data.shape\n', fits_data.shape)
print('fits_data\n', fits_data)

# show fits file 
plt.imshow(fits_data, cmap = 'gray', interpolation = 'None')
plt.show()

# write 16bit monochrome fits file
hdu.writeto(f_name[:-1]+'_changed.'+fileinfo[-1], overwrite =True)