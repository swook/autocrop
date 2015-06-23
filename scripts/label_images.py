#!/usr/bin/env python3

import argparse
import json
import os
import random
import signal
import sys
import time

import cv2   as cv
import numpy as np
import PIL.Image

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Classify images in a directory as good or bad.')
    parser.add_argument('images-path', nargs='?', default='.', help='Path to directory of input images.')
    args = vars(parser.parse_args())

    # Select directory
    global if_path
    if_path = args['images-path']

    # Get list of files
    global files
    valid_suffices = ('png', 'jpg', 'jpeg')
    files = [f for f in os.listdir(if_path) if \
             os.path.isfile('%s/%s' % (if_path, f)) and \
             f.lower().endswith(valid_suffices)]
    if len(files) == 0:
        print('No images to read in: %s' % if_path)
        return

    # Shuffle files to reduce bias due to neighbouring images
    random.shuffle(files)
    #files = sorted(files)

    # Main loop
    global idx, data
    idx = -1
    load()
    next_image()
    if idx >= len(files):
        print('All %d images have been classified.' % len(files))
        idx = 0
    show_image(idx)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        # Show current image
        show_image(idx)

        # Stop if past last image
        if idx >= len(files):
            break

        # Wait for key for 50ms
        k = cv.waitKey(50)

        # Ignore if no key captured
        if k is -1:
            continue

        # If Esc, break loop
        if k == 27:
            break

        # If up button, upvote
        if k == 65362:
            upvote()

        # If down button, downvote
        if k == 65364:
            downvote()

        # If left button, show previous image
        if k == 65361:
            prev_image()

        # If right button, show next image
        if k == 65363:
            next_image()

    cleanup()

def cleanup(signum=signal.SIGTERM, frame=None):
    # Save data
    save()

    # Close all windows
    cv.destroyAllWindows()
    cv.waitKey(0) # Let windows close

    # Exit programme
    sys.exit(0)


def prev_image():
    global idx
    idx -= 1
    idx = 0 if idx < 0 else idx
    show_image(idx)

def next_image():
    global idx, files
    idx += 1

    show_image(idx)

def next_unclassified_image():
    global idx, files

    idx += 1
    while idx < len(files) and files[idx] in data.keys():
        idx += 1

    show_image(idx)

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

    imshow(fpath, cur_img)

    if fpath in data:
        print('[%03d/%03d] Showing %s (%s)' % (idx, len(files), fpath,
            'GOOD' if data[fpath] else 'BAD'))
    else:
        print('[%03d/%03d] Showing %s' % (idx, len(files), fpath))

cv.namedWindow('main', cv.WINDOW_OPENGL)
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

def upvote():
    global cur_img, data, files, idx

    # Set value in data dict
    fn = files[idx]
    data[fn] = 1;

    # Show green tinted image
    img = cur_img
    img[:, :, 1] = 180 # Set G pixels to 180
    imshow(fn, img)

    # Pause then move to next image
    cv.waitKey(500)
    next_unclassified_image()

def downvote():
    global cur_img, data, files, idx

    # Set value in data dict
    fn = files[idx]
    data[fn] = 0;

    # Show red tinted image
    img = cur_img
    img[:, :, 2] = 180 # Set R pixels to 180
    imshow(fn, img)

    # Pause then move to next image
    cv.waitKey(500)
    next_unclassified_image()

data = {} # Is dict with filenames as key
outfile = 'classifications.json'

def load():
    global data, if_path
    out_path = '%s/%s' % (if_path, outfile)
    try:
        with open(out_path, 'r') as f:
            data = json.load(f)
    except:
        data = {}

def save():
    global data, if_path
    out_path = '%s/%s' % (if_path, outfile)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
        print('Saved output to %s' % out_path)

if __name__ == '__main__':
    main()
