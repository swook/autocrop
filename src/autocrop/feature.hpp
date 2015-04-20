#pragma once

#include "opencv2/core.hpp"

#define FEATS_N 21

cv::Mat getFeatureVector(const cv::Mat& img);
