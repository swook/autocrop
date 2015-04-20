#pragma once

#include "opencv2/opencv.hpp"

extern "C" {
#include "vl/generic.h"
#include "vl/slic.h"
}

/**
 * Generates a pattern distinctiveness map
 *
 * 1) [Divide image into 9x9 patches]
 * 2) Perform PCA
 * 3) Project each patch into PCA space
 * 4) Take L1-norm and store to map
 */
cv::Mat _getPatternDistinct(const cv::Mat& img, std::vector<vl_uint32>& segmentation,
                        std::vector<float>& spxl_vars, float var_thresh);

/**
 * Generates a colour distinctiveness map
 *
 * 1) Calculate average colour per SLIC region
 * 2) Calculate sum of euclidean distance between colours
 */
cv::Mat _getColourDistinct(const cv::Mat& img, std::vector<vl_uint32>& segmentation,
                       uint spxl_n);

/**
 * Generates a Gaussian weight map
 *
 * 1) Threshold given distinctiveness map with thresholds in 0:0.1:1
 * 2) Compute centre of mass
 * 3) Place Gaussian with standard deviation 1000 at CoM
 *    (Weight according to threshold)
 */
cv::Mat _getWeightMap(cv::Mat& D);

/**
 * Generates a saliency map using a method from Margolin et al. (2013)
 *
 * 1) Acquire pattern distinctiveness map
 * 2) Acquire colour distinctiveness map
 * 3) Calculate pixelwise multiplication of the two maps
 */
cv::Mat getSaliency(const cv::Mat& img);
