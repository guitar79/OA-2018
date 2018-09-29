from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=111.,llcrnrlat=25.,urcrnrlon=145.,urcrnrlat=50.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)

#m.drawline(llcrnrlon=127,llcrnrlat=34,urcrnrlon=135,urcrnrlat=44,linewidth=2,color='b')

lats = [34,44,44,34,34] 

lons = [127, 127, 135, 135, 127] 

x, y = m(lons, lats) # forgot this line 
m.plot(x, y, 'D-', markersize=0, linewidth=2, color='k', markerfacecolor='b') 

m.drawcoastlines()
m.drawcountries()
m.fillcontinents()
#print country.attributes['name_long'], earth_colors.next()
#plt.annotate('Korea', xy=(1, 1), ha="center")
# draw parallels
m.drawparallels(np.arange(10,90,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
#ax.set_title('Great Circle from New York to London')
plt.show()