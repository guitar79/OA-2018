from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# create new figure, axes instances.
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])

# setup mercator map projection.
m = Basemap(llcrnrlon=111.,llcrnrlat=25.,urcrnrlon=145.,urcrnrlat=50.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)
def plot_rec(bmap, lower_left, upper_left, lower_right, upper_right):
    xs = [127, 127,
          135, 135,
          127, 135,
          127, 135]
    ys = [34, 44,
          34, 44,
          34, 34,
          44, 44]
    m.plot(xs, ys, latlon = True)

m.drawcoastlines()
m.drawcountries()
m.fillcontinents()

# draw parallels
m.drawparallels(np.arange(10,90,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
#ax.set_title('Great Circle from New York to London')
plt.show()