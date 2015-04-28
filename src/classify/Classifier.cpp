#include "Classifier.hpp"
#include "../features/feature.cpp"
using namespace cv;

void Classifier::loadModel(std::string fpath)
{
	model = ml::SVM::load<ml::SVM>(fpath);
}

bool Classifier::classify(const Mat& img)
{
	Mat featVec = getFeatureVector(img, Rect(0, 0, img.cols, img.rows));
	Mat result;
	std::cout << featVec << std::endl;
	model->predict(featVec, result);

	return (int) result.at<float>(0, 0) != 0;
}

