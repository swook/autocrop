#!/usr/bin/env python2

from multiprocessing import cpu_count

from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np

class Trainer:
    clf = None
    svm = None

    def __init__(self):
        self.svm = svm.SVC(kernel='linear', shrinking=True, verbose=False)
        params = {
            'C': np.logspace(-5, 5, num=10), # Range of C values
        }
        self.clf = GridSearchCV(self.svm, params,
            cv      = 10,          # k-fold CV
            n_jobs  = cpu_count(), # Parallelize over CPUs
            verbose = 1,
        )

    def train(self, featMat, persist=True):
        # Preprocess
        scaler = StandardScaler()
        featMat.X = scaler.fit_transform(featMat.X, featMat.y)

        # Save preprocess output
        self.scaler = scaler
        if persist:
            joblib.dump(scaler, 'preprocess.out')

        # Perform CV
        print('Running SVM trainer on %d rows of data with %d features.' % featMat.X.shape)
        self.clf.fit(featMat.X, featMat.y)

        # Save CV output
        self.estimator = self.clf.best_estimator_
        print(self.estimator)
        if persist:
            joblib.dump(self.estimator, 'cv.out')

