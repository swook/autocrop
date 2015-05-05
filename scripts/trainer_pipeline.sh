#!/bin/bash

# Go to where the pipeline script is
cd $(dirname "${BASH_SOURCE[0]}")

BUILD="../build"

# Preprocess Chen images
echo; echo "> Preprocessing Chen images..."
$BUILD/saliency ../datasets/Chen/image --output-dir ../datasets/Chen/image

# Preprocess EarthPorn images
echo; echo "> Preprocessing EarthPorn images..."
$BUILD/saliency ../datasets/EarthPorn --output-dir ../datasets/EarthPorn

# Calculate features
echo; echo "> Calculating features..."
$BUILD/features

# Run trainer
echo; echo "> Running trainer..."
$BUILD/trainer

