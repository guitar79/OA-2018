# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:22:02 2018
@author: park
Use only 16bit monochrome fits file
"""

from PIL import Image
import matplotlib.pyplot as plt

# RGB 각 채널에 해당하는 파일명 입력
R_file = 'mean_image_R.png'
G_file = 'mean_image_G.png'
B_file = 'mean_image_B.png'

# RGB 각 채널별 파일을 8bit grayscale 이미지로 변경
table = [ i/256 for i in range(65536)]

# read each channel file
red = Image.open(R_file)
green = Image.open(G_file)
blue= Image.open(B_file)

# convert 16 bit image to 8 bit 
red = red.point(table, 'L')
green = green.point(table, 'L')
blue = blue.point(table, 'L')

# composite 24bit image using 8bit
RGB_image = Image.merge("RGB", (red, green, blue))

# display images
plt.imshow(RGB_image, interpolation = 'None')
plt.show()

# save images
RGB_image.save('img-out.png','png')