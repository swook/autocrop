#pragma once

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

class Trainer
{
public:
	void init();
	void add(const cv::Mat& img);
	void train();

private:
	cv::Ptr<cv::ml::SVM> model;
	cv::Mat              data;
};

