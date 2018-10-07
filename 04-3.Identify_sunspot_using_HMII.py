# Based on Python 3.5.2 | Anaconda 4.1.1
# Print software information
print('Source : https://github.com/seungwonpark/SunSpotTracker')

#ModuleNotFoundError: No module named 'cv2' 
#conda install -c https://conda.binstar.org/menpo opencv

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle

fname = '20170228_231038_1024_HMII.jpg'

# variables for processing
sun_im = plt.imread(fname)
im_width, im_height = np.shape(sun_im)
im_center_x, im_center_y = im_width/2, im_height/2
print('sun image dimension :', np.shape(sun_im))

# Set figure width to 10 and height to 10
plt.rcParams["figure.figsize"] = [10,10]

# display sun image
print ('*'*80)
print ('display sun image')
plt.imshow(sun_im, cmap = 'gray', interpolation = 'None')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# detect circles of sun from the image
print ('*'*80)
print('detecting circle of sun')
sun_circle = cv2.HoughCircles(sun_im, cv2.HOUGH_GRADIENT, 2, 200)
# convert the (x, y, r) coordinates and radius of the circles to integers
sun_circle = np.round(sun_circle[0,0,:]).astype(int)
print('circle of sun', sun_circle )
'''
if sun_circle is not None:
    # convert the (x, y, r) coordinates and radius of the circles to integers
	sun_circle = np.round(sun_circle[0,0,:]).astype(int)
else :
    print('edge of sun is not detected ')
    sun_circle = [im_center_x, im_center_y, im_center_x-32]
print('circle of sun', sun_circle )
'''    
# Drawing circle on the sun image
print ('*'*80)
print ('Drawing circle on the sun image')
#fig, ax = plt.subplots(figsize=(10, 10))
fig, ax = plt.subplots()
plt.imshow(sun_im, cmap = 'gray', interpolation = 'None')
circle = Circle((sun_circle[0], sun_circle[1]), sun_circle[2], facecolor='none',
            edgecolor=(0, 0.8, 0.8), linewidth=2, alpha=0.5)
ax.add_patch(circle)    
plt.show()  

# Remove black background
# copy image for processing
sun_im_output = sun_im.copy()
X, Y = np.ogrid[0:im_width, 0:im_height]
boundary_width = 60 # variable for removing limb darkening area
outer = (X - sun_circle[0]) ** 2 + (Y - sun_circle[1]) ** 2 > (sun_circle[2] - boundary_width) ** 2 
inner = (X - sun_circle[0]) ** 2 + (Y - sun_circle[1]) ** 2 < (sun_circle[2] - boundary_width) ** 2 
sun_im_output[outer] = 255

print ('*'*80)
print ('after removing black pixels')
print ('boundary width :', boundary_width)
fig, ax = plt.subplots()
plt.imshow(sun_im_output, cmap = 'gray', interpolation = 'None')
circle = Circle((sun_circle[0], sun_circle[1]), sun_circle[2], facecolor='none',
            edgecolor=(0, 0.8, 0.8), linewidth=2, alpha=0.5)
ax.add_patch(circle)    
plt.show()

# Obtaining pixels expected to be sunspots
bright_criterion = 70 # variable for sunspot criterion
sunspot_mask  = np.zeros((im_width, im_height)).astype(bool)
check_pixel = [] # x, y, intensity
for i in range(0, im_width):
    for j in range(0, im_height):
        if(sun_im_output[i,j] < bright_criterion):
            sunspot_mask[i,j] = 0
            check_pixel.append([j, i, sun_im_output[i,j]])
        else : 
            sunspot_mask[i,j] = 1
sun_im_output[sunspot_mask] = 255 #non-sunspot pixels will be white

print ('*'*80)
print ('pixels expected to be sunspots')
print (sun_im_output)
fig, ax = plt.subplots()
plt.imshow(sun_im_output, cmap = 'gray', interpolation = 'None')
circle = Circle((sun_circle[0], sun_circle[1]), sun_circle[2], facecolor='none',
            edgecolor=(0, 0.8, 0.8), linewidth=2, alpha=0.5)
ax.add_patch(circle) 
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

check_pixel = np.asarray(check_pixel)
print ('number of pixels for  for checking if it is sunspot or not', len(check_pixel))
#print (check_pixel)
if len(check_pixel) == 0 : 
    print('There is no sunspot pixel in the image')
else :
    print ('*'*80)
    print ('pixels for checking if it is sunspot or not')
    fig,ax = plt.subplots()
    l = plt.plot(check_pixel[:,0], (im_height-check_pixel[:,1]), 'bo')
    plt.setp(l, markersize=2)
    circle = Circle((sun_circle[0], sun_circle[1]), sun_circle[2], facecolor='none',
                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    #plt.setp(l, markerfacecolor='C0')
    plt.xlim((0, im_width))
    plt.ylim((0, im_height))
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

# Calculation of latitude and longitude of sunspot on sun
def latitude(y):
    return math.asin( (sun_circle[1] - y) / sun_circle[2] )
def longitude(x,y):
    return math.asin( (x - sun_circle[0]) / (sun_circle[2] * math.cos(latitude(y))) )

#######################################################
#### DFS range. Need to determine this taxi radius.####
dx = [0,1,1,1,0,-1,-1,-1]
dy = [1,1,0,-1,-1,-1,0,1]
#######################################################
def dfs(x,y): # DFS(Depth First Search)
    global x_pixel_sum
    global y_pixel_sum
    global num_pixel
    image[x][y] = -1 # mark as already visited.
    x_pixel_sum += x
    y_pixel_sum += y
    
    num_pixel += 1
    for i in range(0,len(dx)):
        if(image[x+dx[i]][y+dy[i]] == 1):
            dfs(x+dx[i],y+dy[i])
#######################################################

# identify sunspot from check_pixel
pixel_num_criterion = 2
sunspot_data = [] #x_average, y_average, num_pixel
image = [[0 for a in range(im_width)] for b in range(im_height)] # make list for checking pixels
for i in check_pixel:
    x = i[0]
    y = i[1]
    image[x][y] = 1
for a in range(0,im_width):
    for b in range(0,im_height):
        if(image[a][b] == 1):
            x_pixel_sum = 0
            y_pixel_sum = 0
            num_pixel = 0
            print('DFS...', a, b)
            dfs(a,b)
            if num_pixel > pixel_num_criterion :
                x_average = x_pixel_sum / num_pixel
                y_average = y_pixel_sum / num_pixel
                latit = latitude(y_average)
                longi = longitude(x_average, y_average)
                latit = math.degrees(latit)
                longi = math.degrees(longi)
                #sunspot_data.append([longi, latit, num_pixel])
                sunspot_data.append([int(x_average), int(y_average), num_pixel])
print ('*'*80)
print ('number of sunspot : ', len(sunspot_data))
print ('sunspot data : ', sunspot_data) #x, y, num_pixel

sunspot_data = np.asarray(sunspot_data)
if len(sunspot_data) == 0 : 
    print('There is no sunspot in the image')
else :
    print ('*'*80)
    print ('draw sunspot coordinate')
    fig,ax = plt.subplots()
    l = plt.plot(sunspot_data[:,0], (im_height-sunspot_data[:,1]), 'ro', markersize=2, markerfacecolor='C0')
    circle = Circle((sun_circle[0], sun_circle[1]), sun_circle[2], facecolor='none',
                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    plt.xlim((0, im_width))
    plt.ylim((0, im_height))    
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    print ('*'*80)
    print ('mark sunspot on the sun')
    # load the sun image
    fig,ax = plt.subplots()
    plt.imshow(sun_im, cmap = 'gray', interpolation = 'None')
    # draw a circle
    circle = Circle((sun_circle[0], sun_circle[1]), sun_circle[2], facecolor='none',
                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)    
    # plot sunspot
    plt.plot(sunspot_data[:,0], (sunspot_data[:,1]), 'ro', markersize=3)
    plt.show()