#!/usr/bin/env python2

import os

import cv2 as cv
import numpy as np
from scipy.stats import pearsonr

from Classifier import *
from FeatMat import *

def main():
    # Initialise classifier
    classifier = Classifier()

    # Go to script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Evaluates accuracy of classifier
    def evaluate_dir(path):
        featMat = FeatMat()
        file_to_feat = featMat.getFeatures(path)
        print('Loaded %d features for %s' % (len(file_to_feat), path))

        classifs = featMat.getClasses(path)

        errs = []
        i = 0
        X = np.ndarray((len(classifs), len(classifs.values()[0])), dtype=float)

        b = 0
        for fname, fclassifs in classifs.iteritems():

            # If any inconsistent classifications
            if len(np.unique(fclassifs)) > 1:
                b += 1

            tru_cls = np.min(fclassifs)
            est_cls = classifier.predictFeats(file_to_feat['%s/%s' % (path, fname)])
            errs.append(tru_cls - est_cls)

            for c, classif in enumerate(fclassifs):
                X[i, c] = classif
            i += 1

        err = float(np.linalg.norm(errs, 2)) # L2-error norm
        print('%.1f%% incorrect' % (100.0 * np.count_nonzero(errs) / len(classifs)))
        print('%.3f L2-error' % (err / len(errs)))

        print('')

        print('%d mismatching classifications' % b)
        print('Pearson\'s coefficient between two classifications: %f' % pearsonr(X[:, 0], X[:, 1])[0])

    evaluate_dir('../datasets/Michael')
    print('')
    evaluate_dir('../datasets/Wookie')

if __name__ == '__main__':
    main()

