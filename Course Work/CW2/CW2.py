""" CS4243 CW 2: Colour spaces, Fourier and Hough transforms
See accompanying powerpoint slides for instructions.

Name: Carissa Ying Geok Teng
Email: e0425113@u.nus.edu
Student ID: A0205190R

Contributors: Neo Yuan Rong Dexter
References: Lecture 3 slides 
"""
import numpy as np
import copy
import cv2
from scipy.linalg import hadamard

import pdb

#BANNED import scipy (Please only use np and cv2 for this assignment)
'''
Helper functions
'''

#Power & Energy functions from CW1, please feel free to use your own code from CW1

def old_am_power(a):
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

def old_am_energy(a):
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
    return pa

#Hadamard helper functions

int2bin = lambda x, n: format(x, 'b').zfill(n)

def horder(b,nn): 
    jj = int2bin(b,nn)
    kk = ''
    for j in range(nn): 
        kk = kk+jj[nn-1-j] 
    
    kkk=np.zeros(nn) 
    kkk[0] = kk[0] 
    for j in range(1,nn):
        kkk[j] = int(kkk[j-1]) ^ int(kk[j]) 
        
    k=0
    for j in range(nn):
        k = k + int(kkk[j]) * 2**(nn-1-j)  

    return int(k)

def ordhad(n): 
    h = hadamard(n)
    hh = hadamard(n)
    nn = np.log2(n)
    for i in range(n):
        k = horder(int(i) , int(nn)) 
        hh[k][:] = h[i][:]

    return hh

'''
End of helper functions
'''

"""Question 3"""
# convert to binary
def threshold(edges, t):
    edges = np.copy(edges)
    edges[edges < t] = 0
    edges[edges >= t] = 1
    return edges

def detect_hough_circles(img, r_min=50, r_max=200, theta=360, detect=1, local_max_area=10):
    """ Detect and draw the circles present in an input image using Hough transformations:
        For example "95103cv.png" should show ...


    Args: img (np.ndarray): Input numpy array for image detection
        r_min = minimum radius of circle for detection
        r_max = maximum radius of circle for detection
        theta = number of degrees rotation from 0deg to 360deg performs a full sweep

    Returns:
        center (Tuple): A python tuple containing the center of (x_c, y_c, r) of the maximum circle
    
    BANNED functions:  cv2.HoughCircles()
    Hint: Use cv2.Canny() to detect the edges of the input image first
    """
    '''YOUR CODE HERE'''    
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    edges = threshold(edges, (np.max(edges)-np.min(edges))/2)
    
    m, n = edges.shape
    r_range = np.arange(r_max-r_min)
    theta_range = np.arange(theta)
    
    # optimisation of circle equation
    c_x = np.transpose(np.tile(r_range+r_min, (len(theta_range), 1))) * np.cos(theta_range*np.pi/180)
    c_y = np.transpose(np.tile(r_range+r_min, (len(theta_range), 1))) * np.sin(theta_range*np.pi/180)
    # map to center
    c_x = (c_x + r_max).astype("uint8")
    c_y = (c_y + r_max).astype("uint8")
    c = np.zeros((2*r_max, 2*r_max, r_max-r_min), dtype="uint8")
    for r in r_range:
        for theta in theta_range:
            c[c_x[r, theta], c_y[r, theta], r] = 1
            
    e = []
    for e_x in range(m):
        for e_y in range(n):
            if edges[e_x, e_y] == 1:
                e.append((e_x, e_y))
    # count
    a = np.zeros((m+2*r_max, n+2*r_max, r_max-r_min), dtype="uint8")
    for e_x, e_y in e:
        a[e_x:e_x+2*r_max, e_y:e_y+2*r_max, :] += c
    
    local_max = []
    for i in range(detect):
        center_x, center_y, r = np.unravel_index(a.argmax(), a.shape)
        local_max.append((center_x-r_max, center_y-r_max, r+r_min))
        for j in range(-local_max_area, local_max_area):
            for k in range(-local_max_area, local_max_area):
                for l in range(-local_max_area, local_max_area):
                    a[center_x+j, center_y+k, r+l] = 0
    '''END OF YOUR CODE'''
    return local_max[0]


"""Question 4 and 5"""
def ButterworthLowPass(m, n, D0, order):
    filter = np.zeros((m, n))
    # normalized cut_off frequency is mapped to real index
    D0 = D0 * min(m,n) / 2
    order = 2 * order
    for i in range(m):
        for j in range(n):
            d = ( (i-m/2)**2 + (j-n/2)**2 )**0.5
            filter[i,j]= 1 / ( 1 + (d/D0)**order )
            
    return filter

def bandpass_filter(img, low=0.1, high=0.4, order=3):
    """ Develop a bandpass filter for an input image:
        Apply on:
         1.) "01a_amusementpark.jpg"
         2.) "JASDF‐1111_.jpg"

    Args: img (np.ndarray): Input numpy array for image filtering with a low and high \
    cut-off frequencies and their order. 
   
    Returns:
        filt_img ([np.ndarray]): Filtered image after bandpass filter of dtype np.uint8
        filt_power (np.float64): Power of the filtered image

    Hint: Use a combination of a LPF and HPF to create a bandreject filter first, then use (1 - bandreject)
    BANNED FUNCTIONS: scipy.signal.butter()

    The unit test provided checks for a scalar value (average pixel value) and the power.
    The main unit test checks for the filt_img matrix value
    """
    m, n = img.shape
    '''YOUR CODE HERE'''
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)
    ft_shift_abs = np.abs(ft_shift)
    ft_shift_phase = np.angle(ft_shift) 
    
    # filtering
    # low is the lowest frequency allowed -> bound of high-pass filter
    # high is the highest frequency allowed -> bound of low-pass filter
    low_pass_f = ButterworthLowPass(m, n, high, order)
    if low == 0:
        high_pass_f = np.ones((m, n))
    else:
        high_pass_f = 1 - ButterworthLowPass(m, n, low, order)
    band_pass_f = np.multiply(low_pass_f, high_pass_f) 
    ft_shift_abs = np.multiply(ft_shift_abs, band_pass_f) 
    
    # rebuild image
    z = np.multiply (ft_shift_abs , np.exp((1j)*(ft_shift_phase)))
    ift_shift = np.fft.ifftshift(z)
    filt_img = np.fft.ifft2(ift_shift)
    filt_img = np.abs(filt_img).astype("uint8")
    
    # power
    filt_power = old_am_power(filt_img)
    '''END OF YOUR CODE'''
        
    return filt_img, filt_power

"""Question 6"""
def idealLowPass(m, n, d_0):
    filter = np.ones((m, n), dtype=np.uint8)
    d_0 = min(m, n) / 2 * d_0
    for i in range(m):
        for j in range(n):
            if ((i-m/2)**2 + (j-n/2)**2)**0.5 >= d_0:
                filter[i,j]= 0
            
    return filter

def bandreject_filter(img, low=0.6, high=0.9):
    """ Develop a Hadamard transform band-reject filter for an input image:
        Apply on:
         1.) "IMG_0358.JPG"
         2.) "06600600u.bmp"

    Args: img (np.ndarray): Input numpy array for image filtering with a low and high \
    cut-off frequencies and their order. 

    Returns:
        filt_img ([np.ndarray]): Filtered image after bandpass filter of dtype np.uint8
        filt_power (np.float64): Power of the filtered image
        
    BANNED FUNCTIONS: scipy.signal.butter() 
    HINT: Use the BPF from Q5 to create a BRF, since BRF = 1 - BPF

    The unit test provided checks for a scalar value (average pixel value) and the power.
    The main unit test checks for the filt_img matrix value
    """    
    m, n = img.shape
    m = 2 ** int(np.log2(min(m, n)))
    '''YOUR CODE HERE'''
    h = ordhad(m)
    w = np.matmul(h, np.matmul(img,h))
    
    # filter
    low_pass_filter = idealLowPass(m*2, m*2, high)[m:, m:]
    if low == 0:
        high_pass_filter = np.ones((m, m), dtype="uint8")
    else:
        high_pass_filter = 1 - idealLowPass(m*2, m*2, low)[m:, m:]
    band_pass_f = np.multiply(low_pass_filter, high_pass_filter)
    band_reject_f = 1 - band_pass_f
    filt_w = np.multiply(w, band_reject_f)
    
    # rebuild image
    filt_img = np.dot(np.matmul(h, np.matmul(filt_w, h)), 1/m**2)
    filt_img = filt_img.astype("uint8")
    
    # power
    filt_power = old_am_power(filt_img)
    '''END OF YOUR CODE'''
    
    return filt_img, filt_power