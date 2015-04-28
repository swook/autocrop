#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
using namespace cv;

#include "../datasets/datasets.hpp"
#include "../saliency/saliency.hpp"
#include "../features/feature.hpp"
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
}

void Trainer::add(const Mat& img, const Rect crop, const int cls)
{
	add(getSaliency(img), getGrad(img), crop, cls);
}

void Trainer::add(const Mat& saliency, const Mat& grad, const Rect crop,
		const int cls)
{
	featMat.addFeatVec(saliency, grad, crop, cls);
}

void Trainer::loadFeatures(std::string fpath)
{
	featMat.load(fpath);
}

void Trainer::train()
{
	auto feats = featMat.getFeatureMatrix();
	auto resps = featMat.getResponseVector();
	auto traindata = ml::TrainData::create(feats, ml::ROW_SAMPLE, resps);
	model->trainAuto(traindata);
}

void Trainer::save(std::string fpath)
{
	model->save(fpath);
}

