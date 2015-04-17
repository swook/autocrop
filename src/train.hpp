#pragma once

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

class Trainer
{
public:
	// Constructor calls init
	Trainer();

	// Initialise, allocate memory for private members
	void init();

	// Add image into training dataset
	void add(const cv::Mat& img);

	// Train model using added datasets
	void train();

	// TODO: Store model to file

private:
	cv::Ptr<cv::ml::SVM> model;

	// Training dataset of size N*FEATS_N where N is number of data points
	// and FEATS_N is the number of features.
	cv::Mat data;
};

