import numpy as np
import matplotlib.pyplot as plt

'''
Q3
'''
def hsv_filter(image_name, set_num, hsvp, points):    
    # get relevant points
    points = points.loc[(points["image"] == image_name) & (points["set_num"] == set_num)].copy()
    # get hue/saturation
    hues = hsvp[:,:,0]
    saturations = hsvp[:,:,1]
    points["h"] = hues[points["y"], points["x"]]
    points["s"] = saturations[points["y"], points["x"]]
    
    hmin = points.min()["h"]
    hmax = points.max()["h"]
    smin = points.min()["s"]
    smax = points.max()["s"]
    
    # mask
    hsv_mask = np.zeros_like(hsvp)
    # 1 if hmin <= h <= hmax and smin <= s <= smax, 0 otherwise
    hsv_mask[(hues >= hmin) & (hues <= hmax) & (saturations >= smin) & (saturations <= smax)] = 1
    
    return hsv_mask, hmin, hmax, smin, smax

'''
Q4
'''

# helper functions
def p1_lbp(img, x, y, w):
    center = img[x, y]
    sample = img[x-1:x+2, y-1:y+2]
    threshold = sample >= center
    return np.sum(threshold * w)

def am_power(a):
    pa = 0.0 
    dim1 = a.shape
    if len(dim1)==2:
        sz = dim1[0] * dim1[1] 
        for i in range(dim1[0]):
            for j in range(dim1[1]):
                pa += a[i,j]**2
    elif len(dim1)==3:
        sz = dim1[0] * dim1[1] * dim1[2]
        for i in range(dim1[0]):
            for j in range(dim1[1]):
                for k in range(dim1[2]):
                    pa += a[i,j,k]**2
    pa = pa / sz
    return pa

# ans
def am_lbp(img):
    # weights
    w = np.array([[1, 2, 4], [128, 0, 8], [64, 32, 16]])
    x, y = img.shape
    lbpres = np.zeros_like(img)
    
    # central pixel
    for i in range(1, x-1):
        for j in range(1, y-1):
            # calculate lbp with radius 3
            lbpres[i, j] = p1_lbp(img, i, j, w)
    
    hist1, bins = np.histogram(lbpres.flatten(),255,[0,255])
    pow1 = am_power(img)

    return lbpres, hist1, pow1

'''
Q5
'''
def am_glcm(img, GL, d, t):
    # getting distance to next pixel
    x_diff = 0
    y_diff = 0
    if t > 0:
        x_diff = -d
    if t < 90:
        y_diff = d
    if t > 90:
        y_diff = -d
        
    ccmm = np.zeros((GL, GL))
    # count pixel pairs
    M, N = img.shape
    for x1 in range(M):
        for y1 in range(N):
            x2 = x1 + x_diff
            y2 = y1 + y_diff
            if (x2 >= 0) and (x2 < M) and (y2 >= 0) and (y2 < N):
                p1 = img[x1, y1]
                p2 = img[x2, y2]
                ccmm[p1, p2] += 1
    
    return ccmm

def glcm_feat(ccmm):
    mxx = np.max(ccmm)
    enrg = np.sum(ccmm**2)
    inrt = 0
    M, N = ccmm.shape
    for i in range(M):
        for j in range(N):
            inrt += (i - j)**2 * ccmm[i,j]
    return mxx, enrg, inrt