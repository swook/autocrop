#include "../constants.hpp"

#include "Classifier.hpp"
#include "../features/feature.cpp"

using namespace cv;

void Classifier::loadModel(std::string fpath)
{
	model = ml::SVM::load<ml::SVM>(fpath);
}

bool Classifier::classify(const Mat& img)
{
	return classify(img, Rect(0, 0, img.cols, img.rows));
}

bool Classifier::classify(const Mat& img, const cv::Rect crop)
{
	Mat saliency = getSaliency(img);
	Mat gradient = getGradient(img);
	return classify(saliency, gradient, crop);
}

bool Classifier::classify(const Mat& saliency, const Mat& gradient)
{
	return classify(saliency, gradient, Rect(0, 0, saliency.cols, saliency.rows));
}

bool Classifier::classify(const Mat& saliency, const Mat& gradient,
                          const cv::Rect crop)
{
	Mat featVec = getFeatureVector(saliency, gradient, crop);
	Mat result;
	model->predict(featVec, result);

	return (int) result.at<float>(0, 0) == GOOD_CROP;
}

