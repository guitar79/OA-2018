# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018

@author: user
"""
#%%
from ccdproc import combine
#%%
#filelist = np.loadtxt('file.list', dtype=bytes).astype(str)
dir_name = '20170220_m35/'
f_name_bias = 'bias-median.fits'
#%%
bias_list = [dir_name+'bias-e-2X2-001.fits', dir_name+'bias-e-2X2-002.fits',\
            dir_name+'bias-e-2X2-003.fits', dir_name+'bias-e-2X2-004.fits',\
            dir_name+'bias-e-2X2-004.fits', dir_name+'bias-e-2X2-006.fits',\
            dir_name+'bias-e-2X2-005.fits', dir_name+'bias-e-2X2-008.fits',\
            dir_name+'bias-e-2X2-006.fits', dir_name+'bias-e-2X2-010.fits']
#%%
bias = combine(bias_list,       # ccdproc does not accept numpy.ndarray, but only python list.
               method='median',         # default is average so I specified median.
               unit='adu')              # unit is required: it's ADU in our case.
print('bias', bias.data.dtype, bias.data.max(), bias.data.min(), bias )
#%%
#save fits file
bias.write(dir_name+f_name_bias, overwrite =True)