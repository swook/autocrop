import os
import re

import numpy as np
import cv2 as cv
import PIL.Image

def filesWithRe(path, regexp):
    files = ['%s/%s' % (path, f) for f in os.listdir(path)]
    return [f for f in files if os.path.isfile(f) and re.match(regexp, f)]

windows = {}
def imshow(name, title, _in):
    img = None

    # Load img if necessary
    if isinstance(_in, str):

        # Try to get rotation data and if set, rotate image correctly
        pil_img = None
        try:
            img       = cv.imread(_in)
            pil_img   = PIL.Image.open(_in)
            rot_code  = pil_img._getexif()[274]
            (h, w, _) = img.shape
            if rot_code == 3:
                img = cv.warpAffine(img, cv.getRotationMatrix2D((w/2., h/2.), 180, 1.), (w, h))
            elif rot_code == 6:
                m = min(h, w)
                img = cv.warpAffine(img, cv.getRotationMatrix2D((m/2., m/2.), -90, 1.), (h, w))
            elif rot_code == 8:
                img = cv.warpAffine(img, cv.getRotationMatrix2D((w/2., w/2.), 90, 1.), (h, w))
            pil_img.close()
        except:
            if pil_img:
                pil_img.close()

    elif isinstance(_in, np.array):
        img = _in

    # Load exception if input argument invalid
    if img is None:
        raise TypeError('Expected file path or image')

    # Create window if necessary
    global windows
    if name not in windows:
        windows[name] = cv.namedWindow('main', cv.WINDOW_OPENGL | cv.WINDOW_KEEPRATIO)

    (h, w, _) = img.shape
    if h < w:
        ratio = 800. / w
    else:
        ratio = 500. / h
    h *= ratio
    w *= ratio

    cv.imshow(name, img)
    cv.setWindowTitle(name, title)
    cv.resizeWindow(name, int(w), int(h))


