import json
import pickle
import util

import numpy as np

class FeatMat:
    X = None
    y = None

    def __init__(self):
        pass

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
        with open(fpath + '.pickle', 'rb') as f:
            features = pickle.load(f)
        return features['fc6']


    # Adds classifications from classifications*.json files found in given path
    def addClasses(self, path, file_to_feat):
        files = util.filesWithRe(path, r'.*\/classifications.*\.json$')
        for fpath in files:
            with open(fpath, 'r') as f:
                classifs = json.load(f)

            for fname, classif in classifs.iteritems():
                try:
                    row = file_to_feat['%s/%s' % (path, fname)]
                    if self.X is None:
                        self.X = np.array([row], dtype=np.float32)
                        self.y = np.array([classif], dtype=np.float32)
                    else:
                        self.X = np.append(self.X, [row], axis=0)
                        self.y = np.append(self.y, classif)
                except:
                    pass

