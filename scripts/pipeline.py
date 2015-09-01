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
    global feats
    cwd = os.getcwd()
    os.chdir('../suitability/')
    feats = Feats(if_path)
    print('- Loaded features for %d files' % len(feats))

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

    # Classify each image and keep suitable ones
    global suitability_scores
    suitable = []
    suitability_scores = []
    for fname in files:
        score = classifier.getScore(feats[fname])
        if score > 0:
            suitable.append(fname)
            suitability_scores.append(score)

    print('- Classified images to retain list of suitable images only')
    print('- %d/%d suitable images left' % (len(suitable), len(files)))
    files = suitable

    # Create list of last shown times
    global last_times
    global selected
    last_times = [time.time()]*len(files)
    selected = [False]*len(files)

    # Main loop
    global num
    num = 0
    find_next_image()
    show_next_image()

    signal.signal(signal.SIGINT, cleanup)

    while True:
        # Wait for key for 50ms
        k = cv.waitKey(50)

        # If no more images left to show, exit
        n_left = len([x for x in selected if not x])
        if n_left == 0:
            break

        # Ignore if no key captured
        if k is -1:
            continue

        # If Esc, break loop
        elif k == 27:
            break

        # If any other button, show previous image
        else:
            next_image()
            print '%d images left to show' % n_left

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
    global idx, files, if_path, feats, last_times, selected, suitability_scores, num

    idx_selected = [i for i, x in enumerate(selected) if x]
    idx_not_selected = [i for i, x in enumerate(selected) if not x]

    now = time.time()

    def dist(idx1, idx2):
        fn1 = files[idx1]
        fn2 = files[idx2]
        return np.linalg.norm(feats[fn1] - feats[fn2])

    def min_dist(idx):
        dists = []
        for j in idx_selected:
            dists.append(dist(idx, j))
        return np.min(dists) if len(dists) > 0 else 0

    def suitability(idx):
        return suitability_scores[idx]

    def get_score(idx):
        l = 0.8 # lambda for weighting
        return l * suitability(idx) + (1. - l) * min_dist(idx)

    def select(idx):
        global num
        selected[idx] = True
        last_times[idx] = now
        num += 1

    # Go through each not selected image and evaluate score
    scores = []
    for j in idx_not_selected:
        scores.append(get_score(j))

    idx = idx_not_selected[np.argmax(scores)]
    select(idx)


def show_next_image():
    global if_path, files, idx, num
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

    # TEMP: Save current image with index since start of slideshow
    if not os.path.isdir('pipeline_out'):
        os.mkdir('pipeline_out')
    cv.imwrite('pipeline_out/%03d.jpg' % num, I)

    show_image(idx, I)

if __name__ == '__main__':
    main()
