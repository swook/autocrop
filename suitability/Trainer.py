from multiprocessing import cpu_count

from sklearn import svm
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

import numpy as np

import config

class Trainer:
    clf = None
    svm = None

    def __init__(self):
        if config.model is 'SVM':
            self.svm = svm.SVC(kernel='linear', shrinking=True, verbose=False)
            params = {
                'C': np.logspace(-5, -1, num=20), # Range of C values
            }
            self.clf = GridSearchCV(self.svm, params,
                cv      = 5,           # k-fold CV
                n_jobs  = cpu_count(), # Parallelize over CPUs
                verbose = 1,
            )
            self.clf_gist = GridSearchCV(self.svm, params,
                cv      = 5,           # k-fold CV
                n_jobs  = cpu_count(), # Parallelize over CPUs
                verbose = 1,
            )

        elif config.model is 'Regression':
            self.clf = LassoCV(
                cv         = 3,
                max_iter   = 2000,
                n_jobs     = cpu_count(),
                verbose    = True,
            )
            self.clf_gist = LassoCV(
                cv         = 3,
                max_iter   = 2000,
                n_jobs     = cpu_count(),
                verbose    = True,
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
        print('Running trainer on %d rows of data with %d features.' % featMat.X.shape)
        self.clf.fit(featMat.X, featMat.y)

        # Save CV output
        if config.model is 'SVM':
            self.estimator = self.clf.best_estimator_
        elif config.model is 'Regression':
            self.estimator = self.clf
        print(self.estimator)

        if persist:
            joblib.dump(self.clf, 'cv.out')

    def train_gist(self, featMat, persist=True):
        # Preprocess
        scaler = StandardScaler()
        featMat.X_gist = scaler.fit_transform(featMat.X_gist, featMat.y)

        # Save preprocess output
        self.scaler_gist = scaler
        if persist:
            joblib.dump(scaler, 'preprocess_gist.out')

        # Perform CV
        print('Running trainer on %d rows of data with %d features.' % featMat.X_gist.shape)
        self.clf_gist.fit(featMat.X_gist, featMat.y)

        # Save CV output
        if config.model is 'SVM':
            self.estimator_gist = self.clf_gist.best_estimator_
        elif config.model is 'Regression':
            self.estimator_gist = self.clf_gist
        print(self.estimator_gist)

        if persist:
            joblib.dump(self.clf_gist, 'cv_gist.out')


