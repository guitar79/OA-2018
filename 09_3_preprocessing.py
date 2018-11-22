# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018

@author: user
"""
#%%
import numpy as np
from ccdproc import CCDData, ccd_process
import astropy.units as u

#%%
#filelist = np.loadtxt('dark.list', dtype=bytes).astype(str)
dir_name = '20170220_m35/'
file_list = [dir_name+'g3035811.fits', dir_name+'g3035812.fits',\
            dir_name+'g3035813.fits', dir_name+'g3035814.fits',\
            dir_name+'g3035815.fits', dir_name+'g3035816.fits',\
            dir_name+'g3035817.fits', dir_name+'g3035818.fits',\
            dir_name+'g3035819.fits', dir_name+'g3035820.fits']

#%%
f_name_bias = 'bias-median.fits'
f_name_dark = 'dark-median.fits'
f_name_flat0 = 'flat0-median.fits'

#%%
# (1) Open master bias, dark, and flat
bias = CCDData.read(dir_name+f_name_bias, unit='adu')
dark = CCDData.read(dir_name+f_name_dark, unit='adu') 
flat0 = CCDData.read(dir_name+f_name_flat0, unit='adu')  # Bias NOT subtracted

#%%
# (2) Reduce each object image separately.
#     Then save it with prefix 'p_' which means "preprocessed"
for objname in file_list :
    obj = CCDData.read(objname, unit='adu')
    reduced = ccd_process(obj,                    # The input image
                          master_bias=bias,       # Master bias
                          dark_frame=dark,        # dark
                          master_flat=flat0,      # non-calibrated flat
                          min_value=30000,        # flat.min() should be 30000
                          exposure_key='exptime', # exposure time keyword
                          exposure_unit=u.s,      # exposure time unit
                          dark_scale=True)        # whether to scale dark frame
    reduced.data = np.array(reduced.data, dtype=np.uint16)
    print('reduced', reduced.data.dtype, reduced.data.min(), reduced.data.max(), reduced)
    reduced.write(objname[:-5]+'_p.fits', overwrite =True)
  