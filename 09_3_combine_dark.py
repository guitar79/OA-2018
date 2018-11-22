# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018

@author: user
"""
#%%
import numpy as np
from astropy.stats import sigma_clip
from ccdproc import CCDData, ccd_process, combine
#%%
#filelist = np.loadtxt('dark.list', dtype=bytes).astype(str)
dir_name = '20170220_m35/'
dark_list = [dir_name+'dark-e-2X2-011.fits', dir_name+'dark-e-2X2-012.fits',\
            dir_name+'dark-e-2X2-013.fits', dir_name+'dark-e-2X2-014.fits',\
            dir_name+'dark-e-2X2-014.fits', dir_name+'dark-e-2X2-016.fits',\
            dir_name+'dark-e-2X2-015.fits', dir_name+'dark-e-2X2-018.fits',\
            dir_name+'dark-e-2X2-016.fits', dir_name+'dark-e-2X2-020.fits']
#%%
f_name_bias = 'bias-median.fits'
f_name_dark0 = 'dark0-median.fits'
f_name_dark = 'dark-median.fits'
#%%
dark0 = combine(dark_list,       # ccdproc does not accept numpy.ndarray, but only python list.
               method='median',         # default is average so I specified median.
               unit='adu')   
#%%
#save fits file
dark0.write(dir_name+f_name_dark0, overwrite =True)
print('dark0', dark0.data.min(), dark0.data.max(), dark0)
# This dark isn't bias subtracted yet, so let's subtract bias:               
#%%
# (1) Open master bias
bias = CCDData.read(dir_name+f_name_bias, unit='adu')
print('bias', bias.data.min(), bias.data.max(), bias.data)
# `unit='adu'` does not necessarily be set if you have header keyword for it.
#%%
# (2) Subtract bias:
dark = ccd_process(dark0, master_bias=bias) 
print('dark', dark.data.min(), dark.data.max(), dark.data)
# This automatically does "dark-bias"
# You can do it by the function "subtract_bias", or just normal pythonic arithmetic.
#%%
# (3) Sigma clip the dark
dark_clip = sigma_clip(dark) 
print('dark_clip', dark_clip.data.min(), dark_clip.data.max(), dark_clip.data)
# by default, astropy.stats.sigma_clip does "3-sigma, 10-iterations" clipping.
# You can tune the parameters by optional keywords (e.g., sigma, iters, ...).
# dark_clip is "numpy masked array". It contains data, mask, and filled_value.
# filled_value is the value which is used to fill the data which are masked (rejected).
# I will change filled_value to be the median value.
#%%
# (4) For rejected pixels, insert the median value
dark_clip.fill_value = np.median(dark_clip.data) 
dark.data = dark_clip.filled() # ~.filled() gives the "data array using filled_value"
print('dark', dark.data.dtype, dark.data.min(), dark.data.max(), dark.data)
#%%
# (5) Save
dark.write(dir_name+f_name_dark, overwrite =True)
