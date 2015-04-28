# Saliency
set(
  SALIENCY_SOURCES
  "util/file.cpp"
  "util/opencv.cpp"
  "util/math.cpp"
  "saliency/SLIC.cpp"
  "saliency/saliency.cpp"
  "features/feature.cpp"
  "features/FeatMat.cpp"
)

add_library(saliency_lib ${SALIENCY_SOURCES})


# Features
set(
  FEATURES_SOURCES
  ${SALIENCY_SOURCES}
  "datasets/datasets.cpp"
)

add_library(features_lib ${FEATURES_SOURCES})

# Classifier
set(
  CLASSIFIER_SOURCES
  ${FEATURES_SOURCES}
  "classify/classify.cpp"
)

add_library(classifier_lib ${CLASSIFIER_SOURCES})


# Trainer
set(
  TRAINER_SOURCES
  ${FEATURES_SOURCES} # TODO: Remove
  "training/train.cpp"
)

add_library(trainer_lib ${TRAINER_SOURCES})


# Autocrop
set(
  AUTOCROP_SOURCES
  ${CLASSIFIER_SOURCES}
  "autocrop/autocrop.cpp"
)

add_library(autocrop_lib ${AUTOCROP_SOURCES})
