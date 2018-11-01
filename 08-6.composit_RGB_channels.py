# -*- coding: utf-8 -*-
"""
Spyder Editor
guitar79@naver.com

conda install tifffile
"""

from PIL import Image
import tifffile as tiff
import pylab as py
#import numpy as np
#폴더명 지정
dir_name = '20161029.NGC2244.all/'

#RGB 각 채널에 해당하는 파일명 입력
R_file = 'mean_image_R.png'
G_file = 'mean_image_R.png'
B_file = 'mean_image_R.png'

#RGB 각 채널별 파일을 8bit grayscale 이미지로 변경
red = Image.open(dir_name+R_file).convert('L')
green = Image.open(dir_name+G_file).convert('L')
blue= Image.open(dir_name+B_file).convert('L')

red = Image.open(dir_name+R_file)
green = Image.open(dir_name+G_file)
blue= Image.open(dir_name+B_file)

a = 
tiff.imsave('new.tiff', a)
#8bit 이미지를 RGB 채널에 넣고 합성
out = Image.merge("RGB", (red, green, blue))

#화면에 파일명을 출력
print("show img-out.png")

#화면에 이미지를 출력 출력
py.imshow(red, aspect='equal')
py.imshow(green, aspect='equal')
py.imshow(blue, aspect='equal')
py.imshow(out, aspect='equal')

#이미지를 저장
out.save(dir_name+"img-out.png",'png')
