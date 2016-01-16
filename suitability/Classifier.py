from sklearn.externals import joblib

from FeatMat import *

class Classifier:
    clf = None

    def __init__(self, trainer=None):
        if trainer:
            if trainer.scaler and trainer.estimator:
                self.preprocess = trainer.scaler
                self.clf = trainer.estimator
            else:
                raise ArgumentError('Provided Trainer is not trained.')
        else:
            self.preprocess = joblib.load('preprocess.out')
            self.clf = joblib.load('cv.out')

    def predictImage(self, imgf):
        feats = FeatMat().getFeature(imgf)
        return self.predictFeats(feats)

    def getScore(self, feats):
        feats = self.preprocess.transform([feats])
        return self.clf.decision_function(feats)[0]

    def predictFeats(self, feats):
        score = self.getScore(feats)
        if score > 0:
            return 1
        else:
            return 0

