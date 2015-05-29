cmake_minimum_required(VERSION 2.8)

set(
  CMAKE_CXX_COMPILER
  "g++"
)

project(sfm)

# Add libraries
find_package(OpenCV REQUIRED core highgui imgproc ml xphoto)
find_package(Boost  REQUIRED program_options system filesystem)

include_directories(${Boost_INCLUDE_DIRS})
#set(Boost_USE_STATIC_LIBS ON)

# OS X specific options
if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -stdlib=libc++")
  set(CMAKE_CXX_LINK_FLAGS "-lc++")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(APPLE)

# Set compile options
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -m64 -DNDEBUG -fopenmp -msse3 -msse4 -mavx"
)

# Add libraries
add_subdirectory(src)

# Binaries
add_executable(saliency   "src/saliency.cpp")
add_executable(features   "src/get_features.cpp")
add_executable(classifier "src/classifier.cpp")
add_executable(trainer    "src/trainer.cpp")
add_executable(autocrop   "src/autocrop.cpp")
add_executable(evaluate   "src/evaluate.cpp")

# Depend on OpenCV and Boost
target_link_libraries(saliency   saliency_lib   -lvl           ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(features   features_lib   -lvl -lcvmatio ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(classifier classifier_lib -lvl           ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(trainer    trainer_lib    -lvl -lcvmatio ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(autocrop   autocrop_lib   -lvl           ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(evaluate   evaluate_lib   -lvl -lcvmatio ${OpenCV_LIBS} ${Boost_LIBRARIES})

