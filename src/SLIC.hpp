#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

extern "C" {
#include "vl/generic.h"
}

void _getSLICSegments(const Mat& img, std::vector<vl_uint32>& segmentation);

float _getSLICVariances(Mat& grey, std::vector<vl_uint32>& segmentation,
                        std::vector<float>& vars);
