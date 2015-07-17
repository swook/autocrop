#!/bin/bash

# Go to where the pipeline script is
cd $(dirname "${BASH_SOURCE[0]}")

BUILD="../build"

# Preprocess Chen images
echo; echo "> Preprocessing Chen images..."
time $BUILD/saliency ../datasets/Chen/image --output-dir ../datasets/Chen/image
notify-send "Autocrop Trainer Pipeline" "Preprocessed Chen\n$(date)"

# Preprocess Reddit images
echo; echo "> Preprocessing Reddit images..."
time $BUILD/saliency ../datasets/Reddit --output-dir ../datasets/Reddit
notify-send "Autocrop Trainer Pipeline" "Preprocessed Reddit\n$(date)"

# Calculate features
echo; echo "> Calculating features..."
time $BUILD/features
notify-send "Autocrop Trainer Pipeline" "Calculated features\n$(date)"

# Run trainer
echo; echo "> Running trainer..."
time $BUILD/trainer
notify-send "Autocrop Trainer Pipeline" "Finished training\n$(date)"

