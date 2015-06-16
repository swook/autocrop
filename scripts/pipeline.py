#!/usr/bin/env python2

import argparse
import os
import random
import signal
import subprocess
import sys
sys.path.append('..')
import time

from suitability.Classifier import *
from suitability.util import *

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
    cwd = os.getcwd()
    os.chdir('../suitability/')
    featMat = FeatMat()
    file_to_feat = featMat.getFeatures(if_path)

    # Initialise classifier
    global classifier
    classifier = Classifier()
    os.chdir(cwd)

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

    signal.signal(signal.SIGINT, cleanup)

    while True:
        # Wait for key for 50ms
        k = cv.waitKey(50)

        # Ignore if no key captured
        if k is -1:
            continue

        # If Esc, break loop
        elif k == 27:
            break

        # If any other button, show previous image
        else:
            next_image()

    cleanup()

tmpfile = 'pipeline_tmp.jpg'
def cleanup(signum=signal.SIGTERM, frame=None):
    # Close all windows
    cv.destroyAllWindows()
    cv.waitKey(0) # Let windows close

    # If tmpfile around, remove
    if os.path.isfile(tmpfile):
        os.remove(tmpfile)

    # Exit programme
    sys.exit(0)

cur_img_path = ''
def show_image(idx, I):
    global cur_img, cur_img_path, if_path, files
    if idx < 0 or idx >= len(files):
        return

    fpath = files[idx]
    new_img_path = '%s/%s' % (if_path, fpath)
    if new_img_path == cur_img_path:
        return
    cur_img_path = new_img_path

    print('- Showing %s [%03d/%03d]' % (fpath, idx, len(files)))
    imshow('main', '%s' % fpath, I)


def next_image():
    global if_path, file_to_feat, cur_img_path, idx, files

    # Get next "suitable" image
    while True:
        idx += 1
        img_path = '%s/%s' % (if_path, files[idx])

        if int(classifier.predictFeats(file_to_feat[img_path])[0]) == 0:
            print('- Skipping unsuitable %s' % img_path)
        else:
            break

    # Automatically crop image
    print('- Cropping %s' % img_path)
    cv.imwrite(tmpfile, imread_rotated(img_path))
    subprocess.check_call(['../build/autocrop', '-i', tmpfile, '-o', tmpfile, '-h'])
    I = cv.imread(tmpfile)

    show_image(idx, I)

if __name__ == '__main__':
    main()
