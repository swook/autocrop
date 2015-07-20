#include <algorithm>

#include <opencv2/core.hpp>
using namespace cv;

#include "autocrop.hpp"
#include "../classify/Classifier.hpp"
#include "../features/feature.hpp"
#include "../saliency/saliency.hpp"
#include "../util/math.hpp"
#include "../util/opencv.hpp"

Mat autocrop(const Mat& in, float w2hrat)
{
	// Get saliency map and sum of values
	Mat saliency = getSaliency(in);

	// Calculate gradient map
	Mat gradient = getGradient(in);

	// Get best crop
	Rect crop = getBestCrop(saliency, gradient, w2hrat);
	std::cout << "Input image is: " << in.size() << std::endl;
	std::cout << "Final crop is: " << crop << std::endl;

	return in(crop);
}

Rect getBestCrop(const Mat& in, float w2hrat)
{
	// Get saliency map and sum of values
	Mat saliency = getSaliency(in);

	// Calculate gradient map
	Mat gradient = getGradient(in);

	return getBestCrop(saliency, gradient, w2hrat);
}

Rect getBestCrop(const Mat& saliency, const Mat& gradient, float w2hrat)
{
	// Initialise classifier
	Classifier classifier;
	classifier.loadModel("Trained_model.yml");

	Candidates candidates = getCropCandidates(classifier, saliency, gradient, w2hrat);
	return candidates[0]->crop;
}

Candidates getCropCandidates(const Classifier& classifier, const Mat& saliency,
	const Mat& gradient, float w2hrat)
{
	bool rat_provided = w2hrat > 1e-5;
	float sum_saliency = sum(saliency)[0];

	// Candidates generation parameters
	const int   MAX_INITIAL_CROP_CANDIDATES = 4000;
	const int   MAX_TOTAL_CROP_CANDIDATES   = 100;
	const float THRESHOLD_REDUCE_FACTOR     = 0.98;

	std::vector<Candidate*> candidates;
	float thresh_content0 = 0.7; // Lower bound of S_content for crop candidates
	if (rat_provided)
		thresh_content0 = w2hrat*saliency.rows/saliency.cols;
	float thresh_content = thresh_content0;

	while (true)
	{
#pragma omp parallel for
		for (int i = 0; i < MAX_INITIAL_CROP_CANDIDATES; i++)
		{
			// Generate single random crop
			Rect crop;
			if (rat_provided) crop = randomCrop(saliency, w2hrat);
			else              crop = randomCrop(saliency);
			Mat cr_saliency = saliency(crop);

			// Calculate content preservation
			float S_content = sum(cr_saliency)[0] / sum_saliency;
			if (S_content < thresh_content) continue;

			// Calculate saliency composition
			Mat cr_gradient = gradient(crop);
			float S_compos = classifier.classifyRaw(cr_saliency, cr_gradient);

			// Add to valid candidates list
#pragma omp critical
			candidates.push_back(new Candidate(crop, S_compos));
		}
		if (candidates.size() > MAX_TOTAL_CROP_CANDIDATES) break;
		thresh_content *= THRESHOLD_REDUCE_FACTOR;
	}
	std::cout << "thresh_content: " << thresh_content0 << " -> " << thresh_content << std::endl;

	const int    C = candidates.size();
	const float fC = (float) C;
	std::cout << C << " valid candidates left." << std::endl;

	// Sort by S_compos in descending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_compos > j->S_compos;
		}
	);

	// Return top 10 crop candidates
	return Candidates(candidates.begin(), candidates.begin() + CROP_CANDS_N);
}


std::ostream& operator<<(std::ostream& os, const Candidate& c)
{
	std::cout << "Crop: "         << c.crop
		  << ", S_compos: "   << c.S_compos
	;
}

