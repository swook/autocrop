#!/usr/bin/env python2

import os
from FeatMat import *
from Trainer import *

def main():
    # Go to script's directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Train on Wookie dataset, evaluate with Michael dataset
    featMat = FeatMat()
    featMat.addFolder('../datasets/Wookie')

    trainer = Trainer()
    trainer.train(featMat)

if __name__ == '__main__':
    main()

