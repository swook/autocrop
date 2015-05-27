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

bool Classifier::classify(const Mat& img) const
{
	return classify(img, Rect(0, 0, img.cols, img.rows));
}

bool Classifier::classify(const Mat& img, const cv::Rect crop) const
{
	Mat saliency = getSaliency(img);
	Mat gradient = getGradient(img);
	return classify(saliency, gradient, crop);
}

bool Classifier::classify(const Mat& saliency, const Mat& gradient) const
{
	return classify(saliency, gradient, Rect(0, 0, saliency.cols, saliency.rows));
}

bool Classifier::classify(const Mat& saliency, const Mat& gradient,
                          const Rect crop) const
{
	Mat featVec = getFeatureVector(saliency, gradient, crop);
	Mat result;
	model->predict(featVec, result);

	return (int) result.at<float>(0, 0) == GOOD_CROP;
}

float Classifier::classifyRaw(const Mat& img) const
{
	Mat saliency = getSaliency(img);
	Mat gradient = getGradient(img);
	return classifyRaw(saliency, gradient, Rect(0, 0, img.cols, img.rows));
}

float Classifier::classifyRaw(const Mat& saliency, const Mat& gradient,
                             const Rect crop) const
{
	Mat featVec = getFeatureVector(saliency, gradient, crop);
	Mat result;
	model->predict(featVec, result, ml::StatModel::RAW_OUTPUT);
	return result.at<float>(0, 0);
}

