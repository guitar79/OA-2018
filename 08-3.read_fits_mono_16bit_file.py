# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park

Use only 16bit monochrome fits file
conda install astropy
"""

from astropy.io import fits
import matplotlib.pyplot as plt

#dir_name = '20181012.M42'
f_name = 'm42-003L.fit'

# read 16bit monochrome fits file
hdu = fits.open(f_name)
print(hdu)
print(dir(hdu))

fits_data = hdu[0].data
print(dir(fits_data))
print(fits_data.shape)
print(fits_data)

# show fits file 
plt.imshow(fits_data, cmap = 'gray', interpolation = 'None')
plt.show()

# write 16bit monochrome fits file
hdu.writeto(f_name[:-4]+'_edited'+f_name[-4:], overwrite =True)