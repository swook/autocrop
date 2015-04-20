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
  "autocrop/feature.cpp"
  "trainer/train.cpp"
)

add_library(trainer_lib ${SALIENCY_SOURCES} ${TRAINER_SOURCES})


# Classifier
set(
  CLASSIFIER_SOURCES
  "autocrop/feature.cpp"
  "classifier/classify.cpp"
)

add_library(classifier_lib ${SALIENCY_SOURCES} ${CLASSIFIER_SOURCES})


# Autocrop
set(
  AUTOCROP_SOURCES
  "autocrop/autocrop.cpp"
)

add_library(autocrop_lib ${SALIENCY_SOURCES} ${CLASSIFIER_SOURCES} ${AUTOCROP_SOURCES})
