#!/usr/bin/env python2

from sklearn import svm
from sklearn.externals import joblib

from FeatMat import *

class Trainer:
    clf = None

    def __init__(self):
        self.clf = svm.SVC(verbose=True)

    def train(self, featMat):
        print('Running SVM trainer on %d rows of data with %d features.' % featMat.X.shape)
        self.clf.fit(featMat.X, featMat.y)

        joblib.dump(self.clf, 'Trained_model.pkl')


def main():
    featMat = FeatMat()
    featMat.addFolder('../datasets/Michael')

    trainer = Trainer()
    trainer.train(featMat)

if __name__ == '__main__':
    main()

