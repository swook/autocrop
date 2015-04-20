#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
using namespace cv;

#include "../datasets/datasets.hpp"
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

	// Matrix with FEATS_N columns
	// NOTE: FEATS_N defined in feature.hpp
	data = Mat(Size(0, FEATS_N), CV_64F);
}

void Trainer::add(const Mat& img)
{
	Mat features = getFeatureVector(img);
	data.push_back(features);

	// TODO: Get class somehow
	int _class = 1;
	// TODO: See if this actually works... print Trainer::data
	data.push_back(_class);
}

void Trainer::addDataset(const ds::DataSet set)
{
	// TODO
}

void Trainer::train()
{
	Mat responses;
	auto traindata = ml::TrainData::create(data, ml::ROW_SAMPLE, responses);
	model->trainAuto(traindata);
}

