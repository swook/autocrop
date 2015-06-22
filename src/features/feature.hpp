#pragma once

#include "opencv2/core.hpp"
#include "FeatMat.hpp"

/**
 * Switch between 3-level Spatial Pyramid of Saliency Maps (SPSM) and 2-level
 * SPSM by enabling the correct define.
 */
//#define spsm3 1
#define spsm2 1

#ifdef spsm3
#define FEATS_N 86
#elif spsm2
#define FEATS_N 22
#endif


/**
 * getFeatureVector constructs a feature vector from a given saliency map and
 * gradient map.
 *
 * - 64 + 16 + 4 + 1 + 1 = 85 features
 *
 * 1) Resize image to 8x8 to average saliency values
 * 2) Store 1/64ths as feature values
 * 3) Average 1/64 values to get feature values for 1/16ths
 * 3) Average 1/16 values to get feature values for 1/4ths
 * 4) Average 1/4 value to get feature value for whole image
 * 5) Sum of all saliency values
 */
cv::Mat getFeatureVector(const cv::Mat& saliency, const cv::Mat& gradient);


/**
 * Convenience methods for providing more or less arguments
 */
cv::Mat getFeatureVector(const cv::Mat& img);
cv::Mat getFeatureVector(const cv::Mat& img, const cv::Rect crop);
cv::Mat getFeatureVector(const cv::Mat& saliency, const cv::Mat& gradient,
	const cv::Rect crop);


/**
 * getGradient gets a gradient map which is used for the boundary simplicity
 * based feature
 */
cv::Mat getGradient(const cv::Mat& img);

