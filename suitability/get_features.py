#!/usr/bin/env python2

import os
import pickle

import caffe
#caffe.set_mode_cpu()
caffe_root = os.path.normpath(os.path.dirname('%s/../../../' % caffe.__file__))

from PIL import Image
import leargist

import numpy as np

import util

def main():
    extractor = FeatureExtractor()
    cache_features(extractor, '../datasets/Wookie')  # Train set
    cache_features(extractor, '../datasets/Michael') # Test set

def cache_features(extractor, path):
    files = util.filesWithRe(path, r'.*\.(jpg|jpeg|png)$')
    for i, fpath in enumerate(files):
        feats = extractor.get_features(fpath)

        with open('%s.pickle' % fpath, 'w') as f:
            pickle.dump(feats, f)

        print('[%d/%d] Stored features for %s' % (i+1, len(files), fpath))

class FeatureExtractor:
    def __init__(self):
        caffe.set_mode_cpu()
        net = caffe.Net(caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt',
                        caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                        caffe.TEST)

        # Input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))

        # Mean pixel
        transformer.set_mean('data', np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))

        # The reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_raw_scale('data', 255)

        # The reference model has channels in BGR order instead of RGB
        transformer.set_channel_swap('data', (2, 1, 0))

        self.net = net
        self.transformer = transformer

    def get_features(self, img_path):
        # Reshape net for single image input
        self.net.blobs['data'].reshape(1, 3, 227, 227)

        img = self.transformer.preprocess('data', caffe.io.load_image(img_path))
        self.net.blobs['data'].data[...] = img

        layer = 'fc7'
        out = self.net.forward(end=layer)

        # Open image for GIST descriptor calculation
        im = Image.open(img_path)

        return {
            'classes': self.net.blobs[layer].data.flatten(),
            'gist': leargist.color_gist(im),
        }

if __name__ == '__main__':
    main()

