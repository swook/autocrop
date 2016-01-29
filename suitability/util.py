#!/usr/bin/env python2

import os
import re
import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/')
import cv2 as cv
import numpy as np
import PIL.Image

def filesWithRe(path, regexp):
    files = ['%s/%s' % (path, f) for f in os.listdir(path)]
    return [f for f in files if os.path.isfile(f) and re.match(regexp, f)]

def imread_rotated(path):
    I = cv.imread(path, cv.IMREAD_UNCHANGED)
    return I # It seems like OpenCV 3 loads images correctly rotated

    with PIL.Image.open(path) as pil_img:
        exif = pil_img._getexif()
        if not exif or 274 not in exif:
            return I
        rot_code  = exif[274]
        print('rot_code: %d' % rot_code)
        (h, w, _) = I.shape
        if rot_code == 3:
            I = cv.warpAffine(I, cv.getRotationMatrix2D((w/2., h/2.), 180, 1.), (w, h))
        elif rot_code == 6:
            m = min(h, w)
            I = cv.warpAffine(I, cv.getRotationMatrix2D((m/2., m/2.), -90, 1.), (h, w))
        elif rot_code == 8:
            I = cv.warpAffine(I, cv.getRotationMatrix2D((w/2., w/2.), 90, 1.), (h, w))
    return I

windows = {}
def imshow(name, title, _in):
    img = None

    # Load img if necessary
    if isinstance(_in, str):
        img = imread_rotated(_in)

    elif isinstance(_in, np.ndarray):
        img = _in

    # Load exception if input argument invalid
    if img is None:
        raise TypeError('Expected file path or image')

    # Create window if necessary
    global windows
    if name not in windows:
        windows[name] = cv.namedWindow(name, cv.WINDOW_OPENGL | cv.WINDOW_KEEPRATIO)

    (h, w, _) = img.shape
    if h < w:
        ratio = 1280. / w
    else:
        ratio = 800. / h
    h *= ratio
    w *= ratio

    img = cv.resize(img, None, fx=ratio, fy=ratio)

    cv.imshow(name, img)
    cv.setWindowTitle(name, title)
    cv.resizeWindow(name, int(w), int(h))


