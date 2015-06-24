#!/usr/bin/env python2

import argparse
import os
import random
import signal
import subprocess
import sys
sys.path.append('..')
import time

from sklearn.preprocessing import normalize

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
    print('- Loaded features for %d files' % len(file_to_feat))

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

    # Create list of last shown times
    global last_times
    last_times = [time.time()]*len(files)

    # Classify each image and keep suitable ones
    suitable = []
    for fname in files:
        img_path = '%s/%s' % (if_path, fname)
        if classifier.predictFeats(file_to_feat[img_path]) == 1:
            suitable.append(fname)

    print('- Classified images to retain list of suitable images only')
    print('- %d/%d suitable images left' % (len(suitable), len(files)))
    files = suitable

    # Main loop
    global idx
    idx = random.randint(0, len(files) - 1)
    show_next_image()

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
    find_next_image()
    show_next_image()

def find_next_image():
    global idx, files, if_path, file_to_feat, last_times

    img_path = '%s/%s' % (if_path, files[idx])
    idx_feat = file_to_feat[img_path]

    # Look through each "suitable" image to find one with maximum distance to
    # last feature vector
    now = time.time()
    n = len(files)
    dists = np.ndarray((1, n), dtype=float)
    novelties = np.ndarray((1, n), dtype=float)
    for i, fname in enumerate(files):
        if i == idx:
            dist = 0.
        else:
            img_path = '%s/%s' % (if_path, fname)
            dist = np.linalg.norm(file_to_feat[img_path] - idx_feat)

        novelty = now - last_times[i]

        dists[0, i] = dist
        novelties[0, i] = novelty

    dists = normalize(dists, axis=1)
    novelties = normalize(novelties, axis=1)

    idx = np.argmax(0.3 * dists + 0.7 * novelties)
    last_times[idx] = time.time()

def show_next_image():
    global if_path, files, idx
    img_path = '%s/%s' % (if_path, files[idx])

    # Automatically crop image (scale down to 1000px width/height max)
    print('- Cropping %s' % img_path)

    I = imread_rotated(img_path)
    h, w, _ = I.shape
    scale = 500. / max(h, w)
    if scale < 1.:
        I = cv.resize(I, None, fx=scale, fy=scale)

    cv.imwrite(tmpfile, I)
    subprocess.check_call('../build/autocrop -h -i %s -o %s -r 0.5625' % (tmpfile, tmpfile), shell=True)
    I = cv.imread(tmpfile)

    show_image(idx, I)

if __name__ == '__main__':
    main()
