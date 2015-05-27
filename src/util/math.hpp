#pragma once

#include <opencv2/core.hpp>

/**
 * Calculates the mean or variance of values in a given list of floats
 */
float mean(std::vector<float>& v);
float var(std::vector<float>& v);


/**
 * Returns a random crop window for a given image.
 */

// This is used to generate "bad" crops for the trainer.
cv::Rect randomCrop(const cv::Mat& img, const cv::Rect good_crop, const float thresh);

// This is used to generate crops of specified aspect ratio
cv::Rect randomCrop(const cv::Mat& img, const float w2hrat = 0.f);


/**
 * Returns an overlap value for two given crops in [0, 1]
 */
float cropOverlap(const cv::Rect crop1, const cv::Rect crop2);

