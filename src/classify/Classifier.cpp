#include "../constants.hpp"

#include "Classifier.hpp"
#include "../features/feature.hpp"
#include "../saliency/saliency.hpp"
#include "../util/opencv.hpp"

using namespace cv;

void Classifier::clear()
{
	model->clear();
}

void Classifier::loadModel(std::string fpath)
{
	model = ml::SVM::load<ml::SVM>(fpath);
}

bool Classifier::_classify(const Mat& featVec) const
{
	Mat result;
	model->predict(featVec, result);
	return (int) result.at<float>(0, 0) == GOOD_CROP;
}

bool Classifier::classify(const Mat& img) const
{
	Mat featVec = getFeatureVector(img);
	return _classify(featVec);
}

bool Classifier::classify(const Mat& img, const cv::Rect crop) const
{
	Mat featVec = getFeatureVector(img, crop);
	return _classify(featVec);
}

bool Classifier::classify(const Mat& saliency, const Mat& gradient) const
{
	Mat featVec = getFeatureVector(saliency, gradient);
	return _classify(featVec);
}

bool Classifier::classify(const Mat& saliency, const Mat& gradient,
                          const Rect crop) const
{
	Mat featVec = getFeatureVector(saliency, gradient, crop);
	return _classify(featVec);
}

float Classifier::_classifyRaw(const Mat& featVec) const
{
	Mat result;
	model->predict(featVec, result, ml::StatModel::RAW_OUTPUT);
	return result.at<float>(0, 0);
}

float Classifier::classifyRaw(const Mat& img) const
{
	Mat featVec = getFeatureVector(img);
	return _classifyRaw(featVec);
}

float Classifier::classifyRaw(const Mat& saliency, const Mat& gradient) const
{
	Mat featVec = getFeatureVector(saliency, gradient);
	return _classifyRaw(featVec);
}

float Classifier::classifyRaw(const Mat& saliency, const Mat& gradient,
                             const Rect crop) const
{
	Mat featVec = getFeatureVector(saliency, gradient, crop);
	return _classifyRaw(featVec);
}

