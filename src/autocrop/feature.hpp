#pragma once

#include "opencv2/core.hpp"

#define FEATS_N 21

cv::Mat getFeatureVector(const cv::Mat& img, const cv::Mat& crop);

cv::Mat getFeatureVector(const cv::Mat& saliency, const cv::Mat& edges,
	const cv::Mat& crop);

cv::Rect getFixedCrop(const cv::Mat& img, const cv::Mat& crop);

cv::Mat getGrad(const cv::Mat& img);

