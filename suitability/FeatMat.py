import json
import pickle
#import multiprocessing

import util

#import pymatlab
#import cv2 as cv
import numpy as np

class FeatMat:
    X = None
    y = None

    def __init__(self):
        pass
        """
        self.matlab = pymatlab.session_factory()
        self.matlab.run('cd ../lib/blur_detection/')
        self.matlab.run('addpath UGM')
        self.matlab.run('addpath feature')
        self.matlab.run('matlabpool open %d' % multiprocessing.cpu_count())

    def __del__(self):
        self.matlab.run('delete(gcp)')
        """

    # Adds folder with features and classification results
    def addFolder(self, path):
        file_to_feat = self.getFeatures(path)
        self.addClasses(path, file_to_feat)
        print('Added folder %s' % path)
        print('There are now %d rows in the feature matrix' % self.X.shape[0])

    # Adds features from .pickle files found in given path
    def getFeatures(self, path):
        file_to_feat = {}

        files = util.filesWithRe(path, r'.*\.pickle$')
        for fpath in files:
            fpath = fpath[:-7] # Strip .pickle
            file_to_feat[fpath] = self.getFeature(fpath)

        return file_to_feat

    # Gets features for one given file
    def getFeature(self, fpath):
        # Get DNN (caffee) features
        feats = []
        with open(fpath + '.pickle', 'rb') as f:
            feats = pickle.load(f)['classes']

        """
        # Get blurriness features
        I = util.imread_rotated(fpath)
        if I.ndim > 1:
            I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
        h, w = I.shape
        scale = 640.0 / max(h, w)
        I = cv.resize(I, None, fx=scale, fy=scale)
        I.astype(numpy.float64)

        # Call blurDetection via matlab
        self.matlab.putvalue('I', I)
        self.matlab.run('I = blurDetection(I);')
        I = self.matlab.getvalue('I')

        cv.imshow('in', I)
        cv.waitKey()
        """

        return feats


    # Gets all available classifications for a give path
    def getClasses(self, path):
        files = util.filesWithRe(path, r'.*\/classifications.*\.json$')
        out = {}
        for fpath in files:
            with open(fpath, 'r') as f:
                classifs = json.load(f)

            good_n = 0
            for fname, classif in classifs.iteritems():
                if classif == 1:
                    good_n += 1
                if fname in out:
                    out[fname].append(classif)
                else:
                    out[fname] = [classif]
            print('%s: %.2f%% are classified as 1.' % (fpath, 100. * good_n / len(classifs)))

        return out


    # Adds classifications from classifications*.json files found in given path
    def addClasses(self, path, file_to_feat):
        classifs = self.getClasses(path)

        for fname, fclassifs in classifs.iteritems():
            classif = 1 if np.mean(fclassifs) >= 0.5 else 0
            row = None
            try:
                row = file_to_feat['%s/%s' % (path, fname)]
            except:
                pass
            if row is None:
                continue

            if self.X is None:
                self.X = np.array([row], dtype=np.float32)
                self.y = np.array([classif], dtype=np.float32)
            else:
                self.X = np.append(self.X, [row], axis=0)
                self.y = np.append(self.y, classif)

