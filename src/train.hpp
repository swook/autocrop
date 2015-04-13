#pragma once

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

class Trainer
{
public:
	void add(cv::Mat& img);
	void train();
};

cv::Mat getFeatureVector(cv::Mat& img);
