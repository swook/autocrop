#!/usr/bin/env python2

import os
import random
import subprocess
import sys
sys.path.append('..')

import numpy as np
import cv2 as cv

from suitability.Classifier import *
from suitability.util import *

def main():
    M = 5
    N = 7
    print('- Output grid of images is %d x %d' % (M, N))
    fnames, imgs, suitable = get_images(M*N)

    # Show all images
    out1 = draw_grid(M, N, imgs)
    cv.imwrite('grid_out1.png', out1)
    print('- Drawn first grid with all images.')

    # Greyscale unsuitable images
    def get_gray(img):
        h, w, _ = img.shape
        gray = np.ndarray((h, w, 1), dtype=np.uint8)
        gray[:, :, 0] = cv.cvtColor(img, cv.COLOR_BGR2GRAY) * 0.3
        gray = np.repeat(gray, 3, 2)
        return gray

    for i, img in enumerate(imgs):
        if not suitable[i]:
            imgs[i] = get_gray(img)

    out2 = draw_grid(M, N, imgs)
    cv.imwrite('grid_out2.png', out2)
    print('- Drawn second grid with unsuitable images grayed out.')

    # Greyscale areas outside of crop regions
    for i, img in enumerate(imgs):
        if suitable[i]:
            print('- Cropping %s' % fnames[i])
            cmd = ['../build/autocrop', '-t', '-h', '-r', '0.5625', '-i', fnames[i]]
            crop_nums = subprocess.check_output(cmd).split()[-4:]
            x, y, w, h = tuple([int(num) for num in crop_nums])
            gray = get_gray(img)
            gray[y:y+h, x:x+w, :] = img[y:y+h, x:x+w, :]
            imgs[i] = gray

    out3 = draw_grid(M, N, imgs)
    cv.imwrite('grid_out3.png', out3)
    print('- Drawn third grid with suitable images cropped.')

def get_images(n):
    if_path = '../datasets/Michael/'

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
    files = ['%s/%s' % (if_path, f) for f in os.listdir(if_path) if \
             os.path.isfile('%s/%s' % (if_path, f)) and \
             f.lower().endswith(valid_suffices)]
    if len(files) == 0:
        print('No images to read in: %s' % if_path)
        cleanup()
    random.shuffle(files)

    # Classify each image and keep suitable ones
    suitable = [False]*len(files)
    for i, fpath in enumerate(files):
        if classifier.predictFeats(file_to_feat[fpath]) == 1:
            suitable[i] = True

    # Get up to n entries
    pairs = sorted(zip(suitable, files))
    good = [(files[i], suitable[i]) for i in range(len(files)) if suitable[i]]
    bad = [(files[i], suitable[i]) for i in range(len(files)) if not suitable[i]]
    random.shuffle(good)
    random.shuffle(bad)
    n_good = n / 4
    n_bad = n - n_good
    pairs = bad[:n_bad] + good[:n_good]
    random.shuffle(pairs)
    files, suitable = zip(*pairs)

    # Load images
    imgs = [None]*len(files)
    for i, fpath in enumerate(files):
        imgs[i] = imread_rotated(fpath)

    return files, imgs, suitable

def draw_grid(M, N, imgs):
    img_w = 200
    img_h = 140
    border = 0

    W = img_w * N + border * (N + 1)
    H = img_h * M + border * (M + 1)

    out = np.zeros((H, W, 3), dtype=np.uint8)

    def resize(img):
        xscal = float(img_w) / img.shape[1]
        yscal = float(img_h) / img.shape[0]
        scale = min(xscal, yscal)
        return cv.resize(img, None, fx=scale, fy=scale)

    def place(img, m, n):
        # Get image dims
        h, w, _ = img.shape

        # Place image
        x0 = (n + 1) * border + n * img_w + img_w / 2 - w / 2
        y0 = (m + 1) * border + m * img_h + img_h / 2 - h / 2
        out[y0:y0+h, x0:x0+w,] = img

    # Resize and place all images
    m = 0
    n = 0
    for i, img in enumerate(imgs):
        img = resize(img)
        place(img, m, n)

        # Increment index
        n += 1
        if n >= N:
            n = 0
            m += 1

    return out

if __name__ == '__main__':
    main()
