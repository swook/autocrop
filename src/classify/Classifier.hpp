#pragma once

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

class Classifier
{
public:
	/**
	 * loadModel loads a trained SVM model from a specified file path
	 */
	void loadModel(std::string fpath);

	/**
	 * classify generates features from a given image and returns a boolean
	 * signifying whether the image is a good crop
	 */
	bool classify(const cv::Mat& img);
private:
	cv::Ptr<cv::ml::SVM> model;
};
