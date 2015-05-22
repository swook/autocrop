#pragma once

#include <opencv2/core.hpp>

/**
 * Calculates the variance of values in a given list of floats
 */
float var(std::vector<float>& v);


/**
 * Returns a random crop window for a given image.
 *
 * This is used to generate "bad" crops for the trainer.
 *
 * TODO: Ensure overlap with "good" crop is not too big
 */
cv::Rect randomCrop(const cv::Mat& img);
cv::Rect randomCrop(const cv::Mat& img, const float w2hrat);

