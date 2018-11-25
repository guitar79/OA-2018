# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018

@author: user
"""
#%%
from glob import glob
import numpy as np
import os
import astropy.units as u
from astropy.stats import sigma_clip
from ccdproc import CCDData, ccd_process, combine
#%%
#filelist = np.loadtxt('file.list', dtype=bytes).astype(str)
dir_name = '20170220_m35/'
f_name_bias = 'bias-median.fits'
f_name_dark0 = 'dark0-median.fits'
f_name_dark = 'dark-median.fits'
f_name_flat0 = 'flat0-median.fits'
f_name_flat = 'flat-median.fits'
#%%
bias_list = sorted(glob(os.path.join(dir_name, 'bias*.*')))
dark_list = sorted(glob(os.path.join(dir_name, 'dark*.*')))
flat_list = sorted(glob(os.path.join(dir_name, 'flat*.*')))
file_list = sorted(glob(os.path.join(dir_name, 'g*.*')))
        
#%%
bias = combine(bias_list,       # ccdproc does not accept numpy.ndarray, but only python list.
               method='median',         # default is average so I specified median.
               unit='adu')              # unit is required: it's ADU in our case.
#print('bias', bias.data.dtype, bias.data.max(), bias.data.min(), bias )
#bias.write(dir_name+f_name_bias, overwrite =True)

#%%
dark0 = combine(dark_list,       # ccdproc does not accept numpy.ndarray, but only python list.
               method='median',         # default is average so I specified median.
               unit='adu')   
dark0.write(dir_name+f_name_dark0, overwrite =True)
print('dark0', dark0.data.min(), dark0.data.max(), dark0)
dark = ccd_process(dark0, master_bias=bias) 
print('dark', dark.data.min(), dark.data.max(), dark.data)
dark_clip = sigma_clip(dark) 
print('dark_clip', dark_clip.data.min(), dark_clip.data.max(), dark_clip.data)
dark_clip.fill_value = np.median(dark_clip.data) 
dark.data = dark_clip.filled() # ~.filled() gives the "data array using filled_value"
print('dark', dark.data.dtype, dark.data.min(), dark.data.max(), dark.data)
#dark.write(dir_name+f_name_dark, overwrite =True)

#%%
flat0 = combine(flat_list,       # ccdproc does not accept numpy.ndarray, but only python list.
               method='median',         # default is average so I specified median.
               unit='adu')  
#flat0.write(dir_name+f_name_flat0, overwrite =True)
print('flat0', flat0.data.min(), flat0.data.max(), flat0)

#%%
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