# -*- coding: utf-8 -*-
"""
Spyder Editor
guitar79@naver.com

"""

from PIL import Image
import pylab as py
#import numpy as np
#폴더명 지정
base_dr = 'C:/KH_data/2018년/G5.동탄고영재학급/180721.COMS/AVHRR/'

#RGB 각 채널에 해당하는 파일명 입력
R_file = "201211281413_m01.bmp"
G_file = "201211281413_m02.bmp"
B_file = "201211281413_m04.bmp"

#RGB 각 채널별 파일을 8bit grayscale 이미지로 변경
red = Image.open(base_dr+R_file).convert('L')
green = Image.open(base_dr+G_file).convert('L')
blue= Image.open(base_dr+B_file).convert('L')

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
out.save(base_dr+"img-out.png",'png')
