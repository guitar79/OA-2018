# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome fits file

conda update --all
conda config --add channels http://ssb.stsci.edu/astroconda
conda install -c astropy ccdproc
"""

from ccdproc import combine
import numpy as np

filelist = np.loadtxt('file.list', dtype=bytes).astype(str)
f_name_output = 'combined1.fits'
bias = combine(filelist.tolist(),       # ccdproc does not accept numpy.ndarray, but only python list.
               #output_file = f_name_output, # I want to save the combined image
               method='median',         # default is average so I specified median.
               unit='adu')              # unit is required: it's ADU in our case.

bias.data = np.array(bias.data, dtype=np.uint16)
bias.write(f_name_output, overwrite =True)

