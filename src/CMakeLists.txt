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
  "util/file.cpp"
  "features/FeatMat.cpp"
  "datasets/Chen.cpp"
  "datasets/Reddit.cpp"

  "autocrop/autocrop.cpp"
  "classify/Classifier.cpp"
)


# Features
set(
  FEATURES_SOURCES
  ${SALIENCY_SOURCES}
)

add_library(features_lib ${FEATURES_SOURCES} ${DATASETS_SOURCES})


# Trainer
set(
  TRAINER_SOURCES
  ${FEATURES_SOURCES}
  "training/Trainer.cpp"
  "features/FeatMat.cpp"
)

add_library(trainer_lib ${TRAINER_SOURCES})


# Classifier
set(
  CLASSIFIER_SOURCES
  ${FEATURES_SOURCES}
  "classify/Classifier.cpp"
)

add_library(classifier_lib ${CLASSIFIER_SOURCES})



# Autocrop
set(
  AUTOCROP_SOURCES
  ${CLASSIFIER_SOURCES}
  "autocrop/autocrop.cpp"
)

add_library(autocrop_lib ${AUTOCROP_SOURCES})


# Evaluate
set(
  EVALUATE_SOURCES
  ${DATASETS_SOURCES}
  ${AUTOCROP_SOURCES}
)

add_library(evaluate_lib ${EVALUATE_SOURCES})

