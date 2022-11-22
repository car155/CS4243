"""Code to run and test your implemented functions.
You do not need to submit this file.

Contributors: Neo Yuan Rong Dexter
"""

import numpy as np
import os
import time
import cv2

from CW2 import *

def wrap_test(func):
    def inner():
        func_name = func.__name__.replace('test_', '')
        try:
            func()
            print('{}: PASSED'.format(func_name))
        except Exception as e:
            print('{}: FAILED, reason: {} ***'.format(func_name, str(e)))
    return inner


@wrap_test
def test_detect_hough_circles():
    img = cv2.imread('circles4.bmp', 0)
    circles = detect_hough_circles(img) # run algo
    
    assert np.allclose(circles, (138, 135, 95), 2), 'detected circle is incorrect'

@wrap_test
def test_bandpass_filter():
    img = cv2.imread("cat.bmp", 0)
    filt_img, filt_power = bandpass_filter(img) # run algo

    print(filt_img.mean(), filt_power)
    assert np.allclose(filt_img.mean(), 15.121170043945312), 'BPF is incorrect'
    assert np.allclose(filt_power, 514.7890472412109), 'BPF power is incorrect'

@wrap_test
def test_bandreject_filter():
    img = cv2.imread("cat.bmp", 0)
    img = cv2.resize(img,(256,256))
    filt_img, filt_power = bandreject_filter(img) # run algo

    print(filt_img.mean(), filt_power)

    assert np.allclose(filt_img.mean(), 89.31590270996094), 'BRF is incorrect'
    assert np.allclose(filt_power, 12267.518081665039), 'BRF power is incorrect'

if __name__ == '__main__':

    t1 = time.time()

    test_detect_hough_circles() #Q3
    test_bandpass_filter() #Q4 and Q5
    test_bandreject_filter() #Q6
    
    t2 = time.time()
    print("Time elapsed:", t2-t1)