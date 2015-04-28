#pragma once

#include "opencv2/core.hpp"
#include "FeatMat.hpp"

#define FEATS_N 23



/**
 * Convenience method for providing source image, not saliency map and edge map.
 *
 * This version is used for cases where only one data point exists per image
 * while datasets should use the other method to reduce calculation of saliency
 * and edge maps.
 */
cv::Mat getFeatureVector(const cv::Mat& img, const cv::Rect crop);


/**
 * getFeatureVector constructs a feature vector from a given saliency map and
 * gradient map.
 *
 * The final feature vector length is:
 *   21 Visual composition
 *    1 Boundary simplicity
 *    1 Content preservation
 *   -----------
 *   23 features
 */
cv::Mat getFeatureVector(const cv::Mat& saliency, const cv::Mat& edges,
	const cv::Rect crop);

/**
 * getGrad gets a gradient map which is used for the boundary simplicity based
 * feature
 */
cv::Mat getGrad(const cv::Mat& img);

