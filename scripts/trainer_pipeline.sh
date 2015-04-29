#!/bin/bash

# Go to where the pipeline script is
cd $(dirname "${BASH_SOURCE[0]}")

BUILD="../build"

# Preprocess Chen images
echo; echo "> Preprocessing Chen images..."
$BUILD/saliency ../datasets/Chen/image --output-dir ../datasets/Chen/image

# Calculate features
echo; echo "> Calculating features..."
$BUILD/features

# Run trainer
echo; echo "> Running trainer..."
$BUILD/trainer

