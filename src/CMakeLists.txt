# Saliency
set(
  SALIENCY_SOURCES
  "util.cpp"
  "saliency/SLIC.cpp"
  "saliency/saliency.cpp"
)

add_library(saliency_lib ${SALIENCY_SOURCES})


# Trainer
set(
  TRAINER_SOURCES
  "datasets/datasets.cpp"
  "features/feature.cpp"
  "training/train.cpp"
)

add_library(trainer_lib ${SALIENCY_SOURCES} ${TRAINER_SOURCES})


# Classifier
set(
  CLASSIFIER_SOURCES
  "features/feature.cpp"
  "classify/classify.cpp"
)

add_library(classifier_lib ${SALIENCY_SOURCES} ${CLASSIFIER_SOURCES})


# Autocrop
set(
  AUTOCROP_SOURCES
  "autocrop/autocrop.cpp"
)

add_library(autocrop_lib ${SALIENCY_SOURCES} ${CLASSIFIER_SOURCES} ${AUTOCROP_SOURCES})
