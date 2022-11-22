import cv2
import pandas as pd
import matplotlib.pyplot as plt
from cw3_util import hsv_filter, am_lbp, am_glcm, glcm_feat




## Q3 example

im_name='m2_samples_563_0.png'
points = pd.read_csv('q3_selected_points_refined.csv')
img = cv2.imread(f"faces/{im_name}")
hsvp = cv2.cvtColor( img , cv2.COLOR_BGR2HSV)

hsv_mask1, hmin1, hmax1, smin1, smax1 = hsv_filter(im_name, 1, hsvp, points)
hsv_mask1, hmin1, hmax1, smin1, smax1 = hsv_filter(im_name, 2, hsvp, points)
hsv_mask1, hmin1, hmax1, smin1, smax1 = hsv_filter(im_name, 3, hsvp, points)


## Q4 example

img = cv2.imread('Gold1.bmp',0)
lbpres, hist2, pow2 = am_lbp(img)
print('Gold1.bmp')
print(pow2)
print(lbpres[532,697], hist2[139])


img = cv2.imread('diag_texture.bmp',0)
lbpres, hist2, pow2 = am_lbp(img)
print('diag_texture.bmp')
print(pow2)
print(lbpres[164, 74], hist2[239])


img = cv2.imread('IMG_0054q.jpg',0)
lbpres, hist2, pow2 = am_lbp(img)
print('IMG_0054q.jpg')
print(pow2)
print(lbpres[295, 631], hist2[124])
print(lbpres[519, 206], hist2[72])


## Q5 example

img = cv2.imread('Gold1.bmp',0)
ccmm = am_glcm( img , GL=256 , d=1 , t=0)
print( glcm_feat( ccmm ))
ccmm = am_glcm( img , GL=256 , d=1 , t=90)
print( glcm_feat( ccmm ))

img = cv2.imread('diag_texture.bmp',0)
ccmm = am_glcm( img , GL=256 , d=1 , t=0)
print( glcm_feat( ccmm ))
ccmm = am_glcm( img , GL=256 , d=1 , t=90)
print( glcm_feat( ccmm ))


img = cv2.imread('IMG_0054q.JPG',0)
ccmm = am_glcm( img , GL=256 , d=1 , t=0)
print( glcm_feat( ccmm ))
ccmm = am_glcm( img , GL=256 , d=1 , t=90)
print( glcm_feat( ccmm ))

img = cv2.imread('IMG_8636q.JPG',0)
ccmm = am_glcm( img , GL=256 , d=1 , t=0)
print( glcm_feat( ccmm ))
ccmm = am_glcm( img , GL=256 , d=1 , t=90)
print( glcm_feat( ccmm ))