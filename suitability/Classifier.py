from sklearn.externals import joblib

from FeatMat import *

class Classifier:
    clf = None

    def __init__(self, trainer=None):
        if trainer:
            if trainer.scaler and trainer.estimator:
                self.preprocess = trainer.scaler
                self.clf = trainer.estimator

		self.preprocess_gist = trainer.scaler_gist
		self.clf_gist = trainer.estimator_gist
            else:
                raise ArgumentError('Provided Trainer is not trained.')
        else:
            self.preprocess = joblib.load('preprocess.out')
            self.clf = joblib.load('cv.out')

            self.preprocess_gist = joblib.load('preprocess_gist.out')
            self.clf_gist = joblib.load('cv_gist.out')

    def predictImage(self, imgf):
        feats = FeatMat().getFeature(imgf)
        return self.predictFeats(feats)

    def getScore(self, feats):
        feats = self.preprocess.transform([feats])
        return self.clf.decision_function(feats)[0]

    def getScore_gist(self, feats):
        feats = self.preprocess_gist.transform([feats])
        return self.clf_gist.decision_function(feats)[0]

    def predictFeats(self, feats):
        score = self.getScore(feats)
        if score > 0:
            return 1
        else:
            return 0

    def predictFeats_gist(self, feats):
        score = self.getScore_gist(feats)
        if score > 0:
            return 1
        else:
            return 0

