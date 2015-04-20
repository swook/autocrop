#pragma once

#include "opencv2/opencv.hpp"

extern "C" {
#include "vl/generic.h"
}

void _getSLICSegments(const cv::Mat& img, std::vector<vl_uint32>& segmentation);

float _getSLICVariances(cv::Mat& grey, std::vector<vl_uint32>& segmentation,
                        std::vector<float>& vars);
