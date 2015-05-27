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
	bool classify(const cv::Mat& img) const;
	bool classify(const cv::Mat& img, const cv::Rect crop) const;
	bool classify(const cv::Mat& saliency, const cv::Mat& gradient) const;
	bool classify(const cv::Mat& saliency, const cv::Mat& gradient,
	              const cv::Rect crop) const;
	float classifyRaw(const cv::Mat& img) const;
	float classifyRaw(const cv::Mat& saliency, const cv::Mat& gradient,
	                  const cv::Rect crop) const;

	/**
	 * clear resets the model to clear memory
	 */
	void clear();

private:
	cv::Ptr<cv::ml::SVM> model;
};
