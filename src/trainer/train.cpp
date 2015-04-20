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
	features  = Mat(Size(FEATS_N, 0), CV_32F);
	responses = Mat(Size(1, 0), CV_32F);
}

void Trainer::add(const Mat& img, const Rect crop, const int cls)
{
	add(getSaliency(img), getGrad(img), crop, cls);
}

void Trainer::add(const Mat& saliency, const Mat& grad, const Rect crop,
		const int cls)
{
	Mat featvec = getFeatureVector(saliency, grad, crop);
	features.push_back(featvec);
	responses.push_back(cls);

	//std::cout << featvec << std::endl;
}

void Trainer::train()
{
	auto traindata = ml::TrainData::create(features, ml::ROW_SAMPLE, responses);
	model->trainAuto(traindata);
}

void Trainer::save()
{
	model->save("autocrop_model.xml");
}

