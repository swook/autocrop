#!/usr/bin/env python2

import os

import numpy as np
from scipy.stats import pearsonr

from Classifier import *
from FeatMat import *
from Trainer import *

class Evaluator:
    name = ""
    logf = None

    train_data = None
    eval_feats = None
    eval_annos = None

    trainer    = None
    classifier = None

    def __init__(self, name):
        self.name = name
        self.init()

    def init(self):
        if self.logf:
            self.logf.close()

        self.train_data = None
        eval_feats      = None
        eval_annos      = None
        self.trainer    = Trainer()

        self.logf = open('evaluate.out', 'a')
        self.logf.write('\n')
        return self

    def log(self, message, section=None):
        if section:
            msg = '[%s|%s] %s' % (self.name, section, message)
        else:
            msg = '[%s] %s' % (self.name, message)

        print(msg)
        self.logf.write(msg + '\n')

    def train_on(self, feats, annotations):
        self.train_data = FeatMat()
        self.train_data.add(feats, annotations)
        return self

    def evaluate_on(self, feats, annotations):
        self.eval_feats = feats
        self.eval_annos = annotations
        return self

    def train(self):
        self.trainer.train(self.train_data, persist=False)
        self.log('Final C: %f' % self.trainer.estimator.C, 'train')
        return self

    def evaluate(self):
        def log(msg):
            self.log(msg, 'eval')

        self.classifier = Classifier(self.trainer)
        print('')

        errs = []
        b = 0
        good_n = 0

        for fname, fclassifs in self.eval_annos.data.iteritems():

            # If any inconsistent classifications
            if len(np.unique(fclassifs)) > 1:
                b += 1

            # Get ground truth class
            classif = np.mean(fclassifs)
            #if classif != 0 and classif != 1:
            #    continue
            tru_cls = 1 if classif >= 0.5 else 0

            if tru_cls == 1:
                good_n += 1

            # Get estimated class
            est_cls = self.classifier.predictFeats(self.eval_feats[fname])

            # Add errors
            errs.append(tru_cls - est_cls)

        log('%d/%d annotations marked with 1' % (good_n, len(self.eval_annos)))

        log('%.1f%% incorrect' % (100.0 * np.count_nonzero(errs) / len(self.eval_annos)))

        l1err = float(np.linalg.norm(errs, 1)) # L2-error norm
        l2err = float(np.linalg.norm(errs, 2)) # L2-error norm
        log('L1-error: %.3f' % (l1err / len(errs)))
        log('L2-error: %.3f' % (l2err / len(errs)))

        return self

    def cleanup(self):
        return self.init()

def main():
    # Reset output file
    if os.path.exists('evaluate.out'):
        os.remove('evaluate.out')

    # Features and Classifications. Dataset_Annotator
    # Michael dataset
    Michael         = Feats('../datasets/Michael')
    Michael_all     = Annotations('../datasets/Michael')
    Michael_Michael = Annotations('../datasets/Michael/classifications_michael.json')
    Michael_Wookie  = Annotations('../datasets/Michael/classifications_wookie.json')
    Michael_Dengxin = Annotations('../datasets/Michael/classifications_dengxin.json')
    Michael_Anphi   = Annotations('../datasets/Michael/classifications_phineasng.json')

    # Wookie dataset
    Wookie         = Feats('../datasets/Wookie')
    Wookie_all     = Annotations('../datasets/Wookie')
    Wookie_Michael = Annotations('../datasets/Wookie/classifications_michael.json')
    Wookie_Wookie  = Annotations('../datasets/Wookie/classifications_wookie.json')
    Wookie_Dengxin = Annotations('../datasets/Wookie/classifications_dengxin.json')
    Wookie_Anphi   = Annotations('../datasets/Wookie/classifications_phineasng.json')

    annotations = [
        (Michael_all,     Wookie_all),
        (Michael_Michael, Wookie_Michael),
        (Michael_Wookie,  Wookie_Wookie),
        (Michael_Dengxin, Wookie_Dengxin),
        (Michael_Anphi,   Wookie_Anphi),
    ]
    for classM, classW in annotations:
        Evaluator('%s -> %s' % (classM.name, classW.name))\
            .train_on(Michael, classM)  \
            .evaluate_on(Wookie, classW) \
            .train().evaluate()

        Evaluator('%s -> %s' % (classW.name, classM.name))\
            .train_on(Wookie, classW) \
            .evaluate_on(Michael, classM)  \
            .train().evaluate()

if __name__ == '__main__':
    main()

