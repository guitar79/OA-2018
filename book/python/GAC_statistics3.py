# -*- coding: utf-8 -*-
import numpy as np
import os
#th = float(input('input threshold: '))
th_list = [11,12,13,14,15,16,17,18,19,20]
for th in th_list:
    th1=str(th)
    #dr = input('input directory: ')
    dt = 'GAC'
    #dr1 = '/media/guitar79/8T/rooknpown/'
    dr1 = 'h:/rooknpown/'
    dr = dr1+'csv'+dt+'/'
    for i in sorted(os.listdir(dr)):
        if i[-4:] == '.csv':
	        f = open(dr+i, 'r').read().split('\n')
	        f = f[1:]
	        f = filter(lambda x: '\t' in x, f)
	        f = np.array(list(map(lambda x: x.split('\t'), f)))
	        totalpix = len(f)
	        g = np.array(list(filter(lambda x: x[1] == 'NaN', f)))
	        nanpix = len(g) 
	        f = np.array(list(filter(lambda x: x[1] != 'NaN' and float(x[1]) > th, f)))
	        statipix = len(f) 
	        if statipix != 0:
	            vsum = np.sum(f[:,1].astype(np.float))
	            vavg = np.mean(f[:,1].astype(np.float))
	            vmed = np.median(f[:,1].astype(np.float))
	            vstd = np.std(f[:,1].astype(np.float))
	            vvar = np.var(f[:,1].astype(np.float))
	            vmax = np.amax(f[:,1].astype(np.float))
	            vmin = np.amin(f[:,1].astype(np.float))
	            with open(dr1+'python_programs/over'+th1+dt+'.txt', 'a') as o:
	                print(i, vsum, vavg, vmed, vstd, vvar, vmax, vmin, totalpix, nanpix, statipix, file=o)
	        else:
	             with open(dr1+'python_programs/over'+th1+dt+'.txt', 'a') as o:
	                 print(i, 0, 0, 0, 0, 0, 0, 0, totalpix, nanpix, statipix, file=o)
