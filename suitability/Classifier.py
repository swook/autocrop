from sklearn.externals import joblib

from FeatMat import *

class Classifier:
    clf = None

    def __init__(self):
        self.clf = joblib.load('Trained_model.pkl')

    def predictImage(self, imgf):
        feats = FeatMat().getFeature(imgf)
        return self.predictFeats(feats)

    def predictFeats(self, feats):
        return self.clf.predict(feats)

