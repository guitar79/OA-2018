#python 
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# create new figure, axes instances.
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])

#loglat = [west,south,east,north]
loglat = [111,25,145,50]
clog = (loglat[2]+loglat[0])/2
clat = (loglat[3]+loglat[1])/2

#map projection.
m = Basemap(llcrnrlon=loglat[0],llcrnrlat=loglat[1],\
            urcrnrlon=loglat[2],urcrnrlat=loglat[3],\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lon_0=clog,lat_0=clat,lat_ts=20.)

m.drawcoastlines()
m.drawcountries()
m.fillcontinents()

# draw parallels
m.drawparallels(np.arange(10,90,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])

plt.show()