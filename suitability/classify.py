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
from util import *

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Classify images in a directory as good or bad.')
    parser.add_argument('images-path', nargs='?', default='.', help='Path to directory of input images.')
    args = vars(parser.parse_args())

    # Select directory
    global if_path
    if_path = args['images-path']

    # Parse in features information
    global feats
    feats = Feats(if_path)

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
        cleanup()

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

    cls = 'GOOD' if classifier.predictFeats(feats[fpath]) == 1 else 'BAD'

    imshow('main', '[%s] %s' % (cls, fpath), cur_img_path)

    print('[%03d/%03d] Showing %s (%s)' % (idx, len(files), fpath, cls))


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

