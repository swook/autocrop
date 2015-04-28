#!/bin/bash

# Go to where the pipeline script is
cd $(dirname "${BASH_SOURCE[0]}")

BUILD="../build"

# Preprocess Chen images
$BUILD/saliency ../datasets/Chen/image --output-dir ../datasets/Chen/image

# Calculate features
$BUILD/features

# Run trainer
$BUILD/trainer

