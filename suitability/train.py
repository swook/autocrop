#!/usr/bin/env python2

from multiprocessing import cpu_count
import os

from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

from numpy import logspace

from FeatMat import *

class Trainer:
    clf = None
    svm = None

    def __init__(self):
        self.svm = svm.SVC(kernel='linear', shrinking=True, verbose=False)
        params = {
            'C': np.logspace(-4, -3, num=50), # Range of C values
        }
        self.clf = GridSearchCV(self.svm, params,
            cv      = 5,           # 5-fold CV
            n_jobs  = cpu_count(), # Parallelize over CPUs
            verbose = 2,
        )

    def train(self, featMat):
        # Preprocess
        scaler = StandardScaler()
        featMat.X = scaler.fit_transform(featMat.X)

        # Save preprocess output
        joblib.dump(scaler, 'preprocess.out')

        # Perform CV
        print('Running SVM trainer on %d rows of data with %d features.' % featMat.X.shape)
        self.clf.fit(featMat.X, featMat.y)

        # Save CV output
        print(self.clf.best_estimator_)
        joblib.dump(self.clf.best_estimator_, 'cv.out')


def main():
    # Go to script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Train on Wookie dataset, evaluate with Michael dataset
    featMat = FeatMat()
    featMat.addFolder('../datasets/Wookie')

    trainer = Trainer()
    trainer.train(featMat)

if __name__ == '__main__':
    main()

