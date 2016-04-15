#!/usr/bin/env python2

from itertools import combinations
import os
from random import randint

import matplotlib
matplotlib.use('PDF')

# Import plotting code
import sys
sys.path.append(os.path.dirname(__file__) + '/../lib/Plotting')
from fastcodeTemplate import generate_graph

import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from Classifier import *
from FeatMat import *
from Trainer import *

TrainingData = {}
Models = {}

class Evaluator:
    name = ""

    train_data = None
    eval_feats = None
    eval_annos = None

    trainer    = None
    classifier = None

    def __init__(self, name):
        self.name = name
        self.init()

    def init(self):
        self.train_data = None
        eval_feats      = None
        eval_annos      = None
        self.trainer    = Trainer()

        return self

    def log(self, message, section=None):
        msg = ''
        if section:
            msg = '[%s|%s] ' % (self.name, section)
        elif self.name:
            msg = '[%s] ' % self.name
        else:
            msg = ''
        msg += message

        print(msg)

        global logf
        logf.write(msg + '\n')

    def train_on(self, feats, annotations):
        global TrainingData
        name = annotations.name
        if name in TrainingData:
            self.train_data = TrainingData[name]
        else:
            self.train_data = FeatMat()
            self.train_data.add(feats, annotations)
            TrainingData[name] = self.train_data
        return self

    def evaluate_on(self, feats, annotations):
        global TrainingData
        name = annotations.name
        if name in TrainingData:
            self.eval_data = TrainingData[name]
        else:
            self.eval_data = FeatMat()
            self.eval_data.add(feats, annotations)
            TrainingData[name] = self.eval_data

        self.eval_feats = feats
        self.eval_annos = annotations
        return self

    def train(self):
        global Models
        name = self.eval_data.name
        if name in Models:
            self.trainer = Models[name]
        else:
            Models[name] = self.trainer
            self.trainer.train(self.train_data, persist=False)
            self.trainer.train_gist(self.train_data, persist=False)
        self.classifier = Classifier(self.trainer)
        return self

    def evaluate(self):
        def log(msg):
            self.log(msg, 'eval')

        print('')

        errs = []
        C = 0                    # Correctly estimated
        I = 0                    # Inconsistent annotations
        A = 0                    # in Agreement (annotations)
        C_A = 0                  # correctly estimated where annotations in agreement
        T = len(self.eval_annos) # Total

        for fname, fclassifs in self.eval_annos.data.iteritems():

            # Get ground truth
            tru = np.median(fclassifs)
            tru_cls = 1 if tru > 0.5 else 0

            # Get prediction
            est = self.classifier.predictFeats(self.eval_feats[fname])
            est_gist = self.classifier.predictFeats_gist(self.eval_feats.gist(fname))
            est_rand = randint(0, 1)
            est_cls = 1 if est > 0.5 else 0

            if tru_cls == est_cls:
                C += 1

            # If any inconsistent classifications
            if len(np.unique(fclassifs)) > 1:
                I += 1
            else:
                A += 1
                if tru_cls == est_cls:
                    C_A += 1

            # Add errors
            errs.append(abs(tru - est))

        A   = float(A)
        C_A = float(C_A)
        T   = float(T)

        global logf
        logf.write('\n')
        if self.eval_annos.n > 1:
            log('%d/%d (%.2f%%) annotations in agreement' % (A, T, A/T*100))
            log('%d/%d (%.2f%%) incorrect for annotations in agreement' % (A-C_A, A, (A-C_A)/A*100))

        log('%d/%d (%.2f%%) incorrect' % (T-C, T, (T-C)/T*100))

        l1err = float(np.linalg.norm(errs, 1)) # L1-error norm
        l2err = float(np.linalg.norm(errs, 2)) # L2-error norm
        log('L1-error: %.3f' % (l1err / len(errs)))
        log('L2-error: %.3f' % (l2err / len(errs)))

        return (T-C)/T*100

    def print_correlations(self, annotations):
        pairs = combinations(annotations, 2)
        Rs = []
        for anno1, anno2 in pairs:
            _, cls1 = zip(*sorted(anno1.data.items()))
            _, cls2 = zip(*sorted(anno2.data.items()))
            cls1 = [x[0] for x in cls1]
            cls2 = [x[0] for x in cls2]
            R, p = pearsonr(cls1, cls2)
            self.log('%s <-> %s: %f %g' % (anno1.name, anno2.name, R, p))
            Rs.append(R)
        self.log('Mean R: %f\n' % np.mean(Rs))

    def plot_PR(self, ofpath, label):
        y_true = self.eval_data.y
        y_score = self.classifier.clf.decision_function(self.eval_data.X)

        # Scale random values to span same range as y_score
        y_score_maxp = np.max(y_score)
        y_score_maxn = -np.min(y_score)
        y_score_span = y_score_maxp + y_score_maxn
        y_rand = np.random.rand(y_true.shape[0], 1) * y_score_span - y_score_maxn

        # Calculate PR (model)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        area = average_precision_score(y_true, y_score)

        x = np.linspace(0, 1, 200)
        y = np.interp(x, np.flipud(recall), np.flipud(precision))

        # Calculate PR (random)
        precision, recall, _ = precision_recall_curve(y_true, y_rand)
        area_rand = average_precision_score(y_true, y_rand)
        y_rand = np.interp(x, np.flipud(recall), np.flipud(precision))

        txtpath = 'PR.out'
        try:
            dat = np.genfromtxt(txtpath, delimiter='\t', names=True,
                                deletechars='', replace_space=False)
        except:
            dat = np.array(x, dtype=[('Recall', float)])

        # Add model PR
        dat = append_fields(dat, '%s (area: %.2f)' % (label, area), y)

        # Add random PR
        dat = append_fields(dat, 'Random classifier (area: %.2f)' % area_rand, y_rand)

        np.savetxt(txtpath, dat, delimiter='\t',
                   header='\t'.join(dat.dtype.names), comments='')

        if ofpath:
            title = 'Precision-Recall curve'
            ylabel = 'Precision'
            generate_graph(txtpath, ofpath, title, ylabel)
        return self

    def cleanup(self):
        return self.init()

logf = None

def main():
    # Reset output files
    if os.path.exists('evaluate.out'):
        os.remove('evaluate.out')
    if os.path.exists('PR.out'):
        os.remove('PR.out')

    global logf
    logf = open('evaluate.out', 'w')
    def log(msg):
        logf.write(msg + '\n')

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

    # Print correlation between annotations
    Evaluator('Annotation correlations (Michael)')\
        .print_correlations([
            Michael_Michael,
            Michael_Wookie,
            Michael_Dengxin,
            Michael_Anphi,
        ])

    Evaluator('Annotation correlations (Wookie)')\
        .print_correlations([
            Wookie_Michael,
            Wookie_Wookie,
            Wookie_Dengxin,
            Wookie_Anphi,
        ])

    # Print errors when training on one and evaluating on other
    annotations = [
        (Michael_Michael, Wookie_Michael, 'Wookie'),
        (Michael_Wookie,  Wookie_Wookie,  'Michael'),
        (Michael_Dengxin, Wookie_Dengxin, 'Dengxin'),
        (Michael_Anphi,   Wookie_Anphi,   'Anphi'),
    ]

    log('Tag format: {Dataset Name}/{Annotator}')
    log('            For example my annotations for the Michael dataset is "Michael/Wookie"')
    log('Format: [Trained on -> Evaluated on]')
    logf.flush()

    Evaluator('Michael/all -> Wookie/all')\
        .train_on(Michael, Michael_all)   \
        .evaluate_on(Wookie, Wookie_all)  \
        .train()                          \
        .plot_PR('PRcurve_michael_wookie.pdf', 'Trained model') \
        .evaluate()

    os.rename('PR.out', 'PR0.out')

    Evaluator('Wookie/all -> Michael/all') \
        .train_on(Wookie, Wookie_all)      \
        .evaluate_on(Michael, Michael_all) \
        .train()                           \
        .plot_PR('PRcurve_wookie_michael.pdf', 'Trained model') \
        .evaluate()

    logf.flush()

    eAlls = []
    eOwns = []

    for classM, classW, name in annotations:
        eAll1 = Evaluator('Michael/all -> %s' % classW.name)\
                    .train_on(Michael, Michael_all)  \
                    .evaluate_on(Wookie, classW) \
                    .train().evaluate()
        eOwn1 = Evaluator('%s -> %s' % (classM.name, classW.name))\
                    .train_on(Michael, classM)  \
                    .evaluate_on(Wookie, classW) \
                    .train().evaluate()
        log('\nIncorrect with all annotations vs just own annotation:')
        log('> %.1f%% vs %.1f%%' % (eAll1, eOwn1))

        eAll2 = Evaluator('Wookie/all -> %s' % classM.name)\
                    .train_on(Wookie, Wookie_all) \
                    .evaluate_on(Michael, classM)  \
                    .train().evaluate()
        eOwn2 = Evaluator('%s -> %s' % (classW.name, classM.name))\
                    .train_on(Wookie, classW) \
                    .evaluate_on(Michael, classM)  \
                    .train().evaluate()
        log('\nIncorrect with all annotations vs just own annotation:')
        log('> %.1f%% vs %.1f%%' % (eAll2, eOwn2))

        log('\nAverage incorrect. All vs own annotations (%s):' % name)
        log('> %.1f%% vs %.1f%%' % (np.mean([eAll1, eAll2]), np.mean([eOwn1, eOwn2])))
        logf.flush()

        eAlls.append(eAll1)
        eOwns.append(eOwn1)
        eAlls.append(eAll2)
        eOwns.append(eOwn2)

    log('\nAverage incorrect (all vs own):')
    log('> %.1f%% vs %.1f%%' % (np.mean(eAlls), np.mean(eOwns)))

if __name__ == '__main__':
    main()

