from mpl_toolkits.basemap import Basemap
#python 
import matplotlib.pyplot as plt
import numpy as np

# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
axisbgc = ax.get_axis_bgcolor()

#loglat = [west,south,east,north]
loglat = [90,0,180,62]
clog = (loglat[2]+loglat[0])/2
clat = (loglat[3]+loglat[1])/2

# map projection.
m = Basemap(llcrnrlon=loglat[0],llcrnrlat=loglat[1],\
    urcrnrlon=loglat[2],urcrnrlat=loglat[3],\
    resolution='l',projection='aea',lon_0=clog,lat_0=clat)
            
m.drawcoastlines()
m.drawcountries()
m.fillcontinents()

# draw parallels
m.drawparallels(np.arange(0,90,10),labels=[1,1,0,0])
# draw meridians
m.drawmeridians(np.arange(-180,180,20),labels=[0,0,1,0])
m.drawmeridians(np.arange(-180,180,10),labels=[0,0,0,1])

plt.show()