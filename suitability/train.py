#!/usr/bin/env python2

import os
from FeatMat import *
from Trainer import *

def main():
    # Go to script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Train on Michael dataset, evaluate with Wookie dataset
    featMat = FeatMat()
    featMat.addFolder('../datasets/Michael')

    trainer = Trainer()
    trainer.train(featMat)
    trainer.train_gist(featMat)

if __name__ == '__main__':
    main()

