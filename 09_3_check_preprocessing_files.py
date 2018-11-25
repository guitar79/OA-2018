# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:00:19 2018

@author: user
"""
#%%
from glob import glob
import os
from ccdproc import CCDData
#%%
dir_name = '20170220_m35/'
file_list = sorted(glob(os.path.join(dir_name, 'g*_p.fits')))

for objname in file_list[0:2] :
    obj = CCDData.read(objname[:-7]+'.fits', unit='adu')
    obj_p = CCDData.read(objname, unit='adu')
   
    print('original', obj.data.dtype, obj.data.min(), obj.data.max(), obj)
    print('preprocessed', obj_p.data.dtype, obj_p.data.min(), obj_p.data.max(), obj_p)
    print('preprocessed file is different from the original file !! : ', obj != obj_p)