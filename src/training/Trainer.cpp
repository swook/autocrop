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

	// Set SVM parameters
	model->setKernel(ml::SVM::LINEAR);
	model->setType  (ml::SVM::C_SVC);

	// Set termination criteria
	TermCriteria termCriteria;
	termCriteria.type     = TermCriteria::COUNT;
	termCriteria.maxCount = 1000;
	termCriteria.epsilon  = 0.01f;
	model->setTermCriteria(termCriteria);

	int  kFold    = 15;   // K-fold Cross-Validation
	bool balanced = true;

	// Set candidate hyperparameter values
	ml::ParamGrid Cgrid      = ml::SVM::getDefaultGrid(ml::SVM::C),
	              gammaGrid  = ml::SVM::getDefaultGrid(ml::SVM::GAMMA),
	              pGrid      = ml::SVM::getDefaultGrid(ml::SVM::P),
	              nuGrid     = ml::SVM::getDefaultGrid(ml::SVM::NU),
	              coeffGrid  = ml::SVM::getDefaultGrid(ml::SVM::COEF),
	              degreeGrid = ml::SVM::getDefaultGrid(ml::SVM::DEGREE)
	;

	// Run cross-validation with SVM
	model->trainAuto(traindata, kFold, Cgrid, gammaGrid, pGrid, nuGrid,
	                 coeffGrid, degreeGrid, balanced);
}

void Trainer::save(std::string fpath)
{
	model->save(fpath);
}

