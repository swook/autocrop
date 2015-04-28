# Saliency
set(
  SALIENCY_SOURCES
  "util/opencv.cpp"
  "util/math.cpp"
  "saliency/SLIC.cpp"
  "saliency/saliency.cpp"
  "features/feature.cpp"
)

add_library(saliency_lib ${SALIENCY_SOURCES} "util/file.cpp")


# Datasets
set(
  DATASETS_SOURCES
  "datasets/datasets.cpp"
  "features/FeatMat.cpp"
  "util/file.cpp"
)


# Features
set(
  FEATURES_SOURCES
  ${SALIENCY_SOURCES}
  "features/feature.cpp"
)

add_library(features_lib ${FEATURES_SOURCES} ${DATASETS_SOURCES})


# Trainer
set(
  TRAINER_SOURCES
  ${FEATURES_SOURCES}
  "training/train.cpp"
  "features/FeatMat.cpp"
)

add_library(trainer_lib ${TRAINER_SOURCES})


# Classifier
set(
  CLASSIFIER_SOURCES
  ${FEATURES_SOURCES}
  "classify/classify.cpp"
)

add_library(classifier_lib ${CLASSIFIER_SOURCES})



# Autocrop
set(
  AUTOCROP_SOURCES
  ${CLASSIFIER_SOURCES}
  "autocrop/autocrop.cpp"
)

add_library(autocrop_lib ${AUTOCROP_SOURCES})

