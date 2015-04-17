set(
  MAIN_SOURCES
  "util.cpp"
  "autocrop.cpp"
  "SLIC.cpp"
  "saliency.cpp"
  "feature.cpp"
  "train.cpp"
  "classify.cpp"
)

add_library(autocrop ${MAIN_SOURCES})

target_link_libraries(autocrop)
