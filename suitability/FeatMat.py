import json
import os
import pickle
from random import randint
import re

import config
import util

import numpy as np

class Feats:
    data = {}
    name = ''
    path = ''

    # Gets features from .pickle files found in given path
    def __init__(self, path):
        self.path = path
        try:
            self.name = re.search(r'\/([^\/]*)\/?$', path).groups()[0]
        except:
            raise ValueError('Invalid path: %s' % path)

        self.data = {}
        for pikpath in util.filesWithRe(path, r'.*\.pickle$'):
            fpath = pikpath[:-7] # Strip .pickle
            with open(pikpath, 'rb') as f:
                self.data[fpath] = pickle.load(f)['classes']

    def __getitem__(self, fname):
        return self.data['%s/%s' % (self.path, fname)]

    def __len__(self):
        return len(self.data)

class Annotations:
    data = {}
    name = ''
    path = ''

    def __init__(self, path, random=False):
        self.path = path

        # Get all annotations in folder or
        files = []
        if os.path.isdir(path):
            files = util.filesWithRe(path, r'.*\/classifications.*\.json$')
            self.name = re.search(r'\/([^\/]*)\/?$', path).groups()[0] + '/all'
        elif os.path.isfile(path) and path.endswith('.json'):
            files = [path]
            self.name = '/'.join(re.search(r'\/([^\/]*)\/classifications_(.*)\.json', path).groups())
        if random:
            self.name = re.search(r'\/([^\/]*)\/?$', path).groups()[0] + '/random'

        self.n = len(files)

        self.data = {}
        for fpath in files:
            with open(fpath, 'r') as f:
                classifs = json.load(f)
                print('Loaded %s' % fpath)
            for fname, classif in classifs.iteritems():
                if random:
                    classif = randint(0, 1)
                if fname in self.data:
                    self.data[fname].append(classif)
                else:
                    self.data[fname] = [classif]

    def __getitem__(self, fname):
        return self.data[fname]

    def __len__(self):
        return len(self.data)


class FeatMat:
    X = None
    y = None
    anno_n = 0

    # Adds folder with features and classification results
    def addFolder(self, path):
        feats = Feats(path)
        annotations = Annotations(path)
        self.add(feats, annotations)
        return self

    # Adds classifications from classifications*.json files found in given path
    def add(self, feats, annotations):
        m = 0
        for fname, fclassifs in annotations.data.iteritems():
            if config.model is 'SVM':
                classif = 1 if np.median(sorted(fclassifs)) > 0.5 else 0
            else:
                classif = np.mean(fclassifs)
            row = None
            try:
                row = feats[fname]
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
            m += 1

        print('Added %d rows (%s), now %d rows' % (m, annotations.name, self.X.shape[0]))
        self.anno_n += annotations.n
        self.name = annotations.name

