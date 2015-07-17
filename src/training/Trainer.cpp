#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
using namespace cv;

#include "../datasets/datasets.hpp"
#include "../saliency/saliency.hpp"
#include "../features/feature.hpp"
#include "Trainer.hpp"

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
	add(getSaliency(img), getGradient(img), crop, cls);
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

	std::cout << "> Training with " << feats.rows << " rows of data." << std::endl;
	std::cout << "> Training with " << feats.cols << " features." << std::endl;

	// Split training data into training and testing subset
	//traindata->setTrainTestSplitRatio(.80f, true);

	// Set SVM parameters
	model->setType(ml::SVM::C_SVC);
	model->setKernel(ml::SVM::LINEAR);

	model->setC(10.f);

	int  kFold    = 20;   // K-fold Cross-Validation
	bool balanced = true;

	// Set candidate hyperparameter values
	ml::ParamGrid CGrid      = ml::SVM::getDefaultGrid(ml::SVM::C),
	              gammaGrid  = ml::SVM::getDefaultGrid(ml::SVM::GAMMA),
	              pGrid      = ml::SVM::getDefaultGrid(ml::SVM::P),
	              nuGrid     = ml::SVM::getDefaultGrid(ml::SVM::NU),
	              coeffGrid  = ml::SVM::getDefaultGrid(ml::SVM::COEF),
	              degreeGrid = ml::SVM::getDefaultGrid(ml::SVM::DEGREE)
	;

	CGrid.logStep = 1.1f;
	CGrid.minVal = 1e-4;
	CGrid.maxVal = 1e4;

	// Run cross-validation with SVM
	model->trainAuto(traindata, kFold, CGrid, gammaGrid, pGrid, nuGrid,
			 coeffGrid, degreeGrid, balanced);

	float err;

	// Calculate error on whole dataset
	err = model->calcError(traindata, false, noArray());
	std::cout << "> Training error on whole dataset: " << err << std::endl;

	// Calculate error on test data subset only
	//err = model->calcError(traindata, true, noArray());
	//std::cout << "> Training error on test data subset: " << err << std::endl;
}

void Trainer::save(std::string fpath)
{
	model->save(fpath);
}

