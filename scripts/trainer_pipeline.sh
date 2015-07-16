#!/bin/bash

# Go to where the pipeline script is
cd $(dirname "${BASH_SOURCE[0]}")

BUILD="../build"

# Preprocess Chen images
echo; echo "> Preprocessing Chen images..."
time $BUILD/saliency ../datasets/Chen/image --output-dir ../datasets/Chen/image

# Preprocess Reddit images
echo; echo "> Preprocessing Reddit images..."
time $BUILD/saliency ../datasets/Reddit --output-dir ../datasets/Reddit

# Calculate features
echo; echo "> Calculating features..."
time $BUILD/features

# Run trainer
echo; echo "> Running trainer..."
time $BUILD/trainer

