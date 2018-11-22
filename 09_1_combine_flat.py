# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018

@author: user
"""
#%%
from ccdproc import CCDData, ccd_process, combine
import astropy.units as u
#%%
#filelist = np.loadtxt('dark.list', dtype=bytes).astype(str)
dir_name = '20170220_m35/'
flat_list = [dir_name+'flatB-e-2X2-001.fits', dir_name+'flatB-e-2X2-001.fits',\
            dir_name+'flatB-e-2X2-001.fits', dir_name+'flatB-e-2X2-001.fits',\
            dir_name+'flatB-e-2X2-001.fits']
#%%
f_name_bias = 'bias-median.fits'
f_name_dark = 'dark-median.fits'
f_name_flat0 = 'flat0-median.fits'
f_name_flat = 'flat-median.fits'

#%%
flat0 = combine(flat_list,       # ccdproc does not accept numpy.ndarray, but only python list.
               method='median',         # default is average so I specified median.
               unit='adu')  

#%%
# write fits file
flat0.write(dir_name+f_name_flat0, overwrite =True)
print('flat0', flat0.data.min(), flat0.data.max(), flat0)
# This dark isn't bias subtracted yet, so let's subtract bias:               

#%%
# (1) Open master bias and dark
bias = CCDData.read(dir_name+f_name_bias, unit='adu')
dark = CCDData.read(dir_name+f_name_dark, unit='adu') 

#%%
# (2) Subtract bias and dark
flat = ccd_process(flat0,                  # The input image (median combined flat)
                   master_bias=bias,       # Master bias
                   dark_frame=dark,        # dark
                   dark_exposure=140 * u.s, # exposure time of dark
                   data_exposure=5.84 * u.s, # exposure time of input image (flat)
                   dark_scale=True)        # whether to scale dark frame
print('flat', flat.data.dtype, flat.data.min(), flat.data.max(), flat)

#%%
# (3) Save
flat.write(dir_name+f_name_flat, overwrite =True)