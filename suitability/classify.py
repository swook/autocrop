#!/usr/bin/env python2

import argparse
import os
import random
import signal
import sys

import cv2 as cv
import PIL.Image

from Classifier import *
from FeatMat import *
import util

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Classify images in a directory as good or bad.')
    parser.add_argument('images-path', nargs='?', default='.', help='Path to directory of input images.')
    args = vars(parser.parse_args())

    # Select directory
    global if_path
    if_path = args['images-path']

    # Parse in features information
    global file_to_feat
    featMat = FeatMat()
    file_to_feat = featMat.getFeatures(if_path)

    # Initialise classifier
    global classifier
    classifier = Classifier()

    # Get list of files
    global files
    valid_suffices = ('png', 'jpg', 'jpeg')
    files = [f for f in os.listdir(if_path) if \
             os.path.isfile('%s/%s' % (if_path, f)) and \
             f.lower().endswith(valid_suffices)]
    if len(files) == 0:
        print('No images to read in: %s' % if_path)
        return

    random.shuffle(files)

    # Main loop
    global idx
    idx = -1
    next_image()
    show_image(idx)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        # Show current image
        show_image(idx)

        # Wait for key for 50ms
        k = cv.waitKey(50)

        # Ignore if no key captured
        if k is -1:
            continue

        # If Esc, break loop
        if k == 27:
            break

        # If left button, show previous image
        if k == 65361:
            prev_image()

        # If right button, show next image
        if k == 65363:
            next_image()

    cleanup()

def cleanup(signum=signal.SIGTERM, frame=None):
    # Close all windows
    cv.destroyAllWindows()
    cv.waitKey(0) # Let windows close

    # Exit programme
    sys.exit(0)

cur_img_path = ''
def show_image(idx):
    global cur_img, cur_img_path, if_path, files
    if idx < 0 or idx >= len(files):
        return

    fpath = files[idx]
    new_img_path = '%s/%s' % (if_path, fpath)
    if new_img_path == cur_img_path:
        return

    cur_img_path = new_img_path
    cur_img = cv.imread(cur_img_path)

    # Try to get rotation data and if set, rotate image correctly
    pil_img = None
    try:
        pil_img   = PIL.Image.open(cur_img_path)
        rot_code  = pil_img._getexif()[274]
        (h, w, _) = cur_img.shape
        if rot_code == 3:
            cur_img = cv.warpAffine(cur_img, cv.getRotationMatrix2D((w/2., h/2.), 180, 1.), (w, h))
        elif rot_code == 6:
            m = min(h, w)
            cur_img = cv.warpAffine(cur_img, cv.getRotationMatrix2D((m/2., m/2.), -90, 1.), (h, w))
        elif rot_code == 8:
            cur_img = cv.warpAffine(cur_img, cv.getRotationMatrix2D((w/2., w/2.), 90, 1.), (h, w))
        pil_img.close()
    except:
        if pil_img:
            pil_img.close()

    cls = 'GOOD' if int(classifier.predictFeats(file_to_feat[cur_img_path])[0]) == 1 else 'BAD'

    imshow('[%s] %s' % (cls, fpath), cur_img)

    print('[%03d/%03d] Showing %s (%s)' % (idx, len(files), fpath, cls))

cv.namedWindow('main', cv.WINDOW_OPENGL | cv.WINDOW_KEEPRATIO)
def imshow(name, img):
    # Make it 800px wide on major axis
    (h, w, _) = img.shape
    if h < w:
        ratio = 800. / w
    else:
        ratio = 500. / h
    h *= ratio
    w *= ratio

    cv.imshow('main', img)
    cv.setWindowTitle('main', name)
    cv.resizeWindow('main', int(w), int(h))

def prev_image():
    global idx
    idx -= 1
    idx = 0 if idx < 0 else idx
    show_image(idx)

def next_image():
    global idx, files
    idx += 1

    show_image(idx)

if __name__ == '__main__':
    main()

