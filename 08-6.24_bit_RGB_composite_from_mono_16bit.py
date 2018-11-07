# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome png files
"""

from PIL import Image
import matplotlib.pyplot as plt

# read each channel file
red = Image.open('mean_image_R.png')
green = Image.open('mean_image_G.png')
blue= Image.open('mean_image_B.png')
print('red :', red)
print('green :', green)
print('blue :', blue)

# making list for converting 16bit to 8bit
table = [ i/256 for i in range(65536)]

# convert 16 bit image to 8 bit 
red = red.point(table, 'L')
green = green.point(table, 'L')
blue = blue.point(table, 'L')
print('red :', red)
print('green :', green)
print('blue :', blue)

# composite 24bit image using 8bit
RGB_image = Image.merge("RGB", (red, green, blue))
print('RGB_image :', RGB_image)

# display images
plt.imshow(RGB_image, interpolation = 'None')
plt.show()

# save images
RGB_image.save('RGB_24bit.png','png')