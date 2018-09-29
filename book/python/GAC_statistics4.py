# -*- coding: utf-8 -*-
import numpy as np
import os
#th = float(input('input threshold: '))
th_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65]
#dr = input('input directory: ')
dt = 'GAC'
#dr1 = '/media/guitar79/8T/rooknpown/'
dr1 = 'h:/rooknpown/'
dr = dr1+'csv'+dt+'/'
for i in sorted(os.listdir(dr)):
    for th in th_list:
        if i[-4:] == '.csv':
            f = open(dr+i, 'r').read().split('\n')
            f = f[1:]
            f = filter(lambda x: '\t' in x, f)
            f = np.array(list(map(lambda x: x.split('\t'), f)))
            totalpix = len(f)
            g = np.array(list(filter(lambda x: x[1] == 'NaN', f)))
            nanpix = len(g)
            th1=str(th)
            f = np.array(list(filter(lambda x: x[1] != 'NaN' and float(x[1]) >= th and float(x[1]) < th+5, f)))
            statipix = len(f) 
            if statipix != 0:
                vsum = np.sum(f[:,1].astype(np.float))
                vavg = np.mean(f[:,1].astype(np.float))
                vmed = np.median(f[:,1].astype(np.float))
                vstd = np.std(f[:,1].astype(np.float))
                vvar = np.var(f[:,1].astype(np.float))
                vmax = np.amax(f[:,1].astype(np.float))
                vmin = np.amin(f[:,1].astype(np.float))
                with open(dr1+'python_programs/statistics'+dt+'.txt', 'a') as o:
                    print(i, th1+'--',vsum, vavg, vmed, vstd, vvar, vmax, vmin, totalpix, nanpix, statipix, file=o)
            else:
                with open(dr1+'python_programs/statistics'+dt+'.txt', 'a') as o:
                    print(i, th1+'--',0, 0, 0, 0, 0, 0, 0, totalpix, nanpix, statipix, file=o)
