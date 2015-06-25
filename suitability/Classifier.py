from sklearn.externals import joblib

from FeatMat import *

class Classifier:
    clf = None

    def __init__(self):
        self.preprocess = joblib.load('preprocess.out')
        self.clf = joblib.load('cv.out')

    def predictImage(self, imgf):
        feats = FeatMat().getFeature(imgf)
        return self.predictFeats(feats)

    def predictFeats(self, feats):
        feats = self.preprocess.transform(feats)
        return int(self.clf.predict(feats)[0])

