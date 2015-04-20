#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
using namespace cv;

#include "../datasets/datasets.hpp"
#include "../saliency/saliency.hpp"
#include "../autocrop/feature.hpp"
#include "train.hpp"

Trainer::Trainer()
{
	init();
}

void Trainer::init()
{
	// Reset or create SVM model
	if (model)  model->clear();
	else        model = ml::SVM::create();

	// Matrix with FEATS_N+1 columns
	// NOTE: FEATS_N defined in feature.hpp
	data = Mat(Size(0, FEATS_N + 1), CV_32F);
}

void Trainer::add(const Mat& img, const Mat& crop, const int cls)
{
	add(getSaliency(img), getGrad(img), crop, cls);
}

void Trainer::add(const Mat& saliency, const Mat& grad, const Mat& crop,
		const int cls)
{
	try
	{
		Mat features = getFeatureVector(saliency, grad, crop);
		features.at<float>(0, FEATS_N) = cls;
		data.push_back(features);
		std::cout << features << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cout << "Invalid crop for given image" << std::endl;
	}
}

void Trainer::train()
{
	Mat responses;
	auto traindata = ml::TrainData::create(data, ml::ROW_SAMPLE, responses);
	model->trainAuto(traindata);
}

