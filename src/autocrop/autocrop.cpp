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
#if FANG
	classifier.loadModel("Trained_model_Fang.yml");
#else
	classifier.loadModel("Trained_model.yml");
#endif

	Candidates candidates = getCropCandidates(classifier, saliency, gradient, w2hrat);
	return candidates[0]->crop;
}

Candidates getCropCandidates(const Classifier& classifier, const Mat& saliency,
	const Mat& gradient, float w2hrat)
{
	bool rat_provided = w2hrat > 1e-5;
	float sum_saliency = sum(saliency)[0];

	// Candidates generation parameters
#if FANG
	const int   MAX_INITIAL_CROP_CANDIDATES = 10000;
	const int   MAX_TOTAL_CROP_CANDIDATES   = 100;
	float thresh_content0 = 0.5;
#else
	const int   MAX_INITIAL_CROP_CANDIDATES = 4000;
	const int   MAX_TOTAL_CROP_CANDIDATES   = 100;
	const float THRESHOLD_REDUCE_FACTOR     = 0.98;
	float thresh_content0 = 0.7; // Lower bound of S_content for crop candidates
#endif

	std::vector<Candidate*> candidates;
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
#if FANG
			if (S_content < thresh_content0) continue;
#else
			if (S_content < thresh_content) continue;
#endif

			// Calculate saliency composition
			Mat cr_gradient = gradient(crop);
			float S_compos = classifier.classifyRaw(cr_saliency, cr_gradient);

#if FANG
			// Add boundary simplicity
			float S_boundary = 0.0f;
			int b = 2, w = crop.width, h = crop.height;
			S_boundary += mean(cr_gradient(Rect(0, 0, w, b)))[0];
			S_boundary += mean(cr_gradient(Rect(0, h-1-b, w, b)))[0];
			S_boundary += mean(cr_gradient(Rect(0, 0, b, h)))[0];
			S_boundary += mean(cr_gradient(Rect(w-1-b, 0, b, h)))[0];
			S_boundary /= 4.0f;
#pragma omp critical
			candidates.push_back(new Candidate{crop, S_compos, S_boundary});
#else

			// Add to valid candidates list
#pragma omp critical
			candidates.push_back(new Candidate{crop, S_compos});
#endif
		}
		if (candidates.size() > MAX_TOTAL_CROP_CANDIDATES) break;
#if !FANG
		thresh_content *= THRESHOLD_REDUCE_FACTOR;
#endif
	}
#if !FANG
	std::cout << "thresh_content: " << thresh_content0 << " -> " << thresh_content << std::endl;
#endif

	const int    C = candidates.size();
	const float fC = (float) C;
	std::cout << C << " valid candidates left." << std::endl;

	// Sort by S_compos in ascending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_compos < j->S_compos;
		}
	);

#if FANG
	// Write R_compos
	for (int i = 0; i < C; ++i) candidates[i]->R_compos = (float) i / fC;

	// Sort by S_boundary in ascending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_boundary < j->S_boundary;
		}
	);

	// Write R_boundary and S_final
	for (int i = 0; i < C; ++i) {
		const auto& candidate = *candidates[i];
		candidates[i]->R_boundary = (float) i / fC;
		candidates[i]->S_final = 5.0f * candidate.R_compos +
			candidate.R_boundary;
	}

	// Sort by S_final in ascending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_final < j->S_final;
		}
	);
#endif

	// Return top 10 crop candidates
	return Candidates(candidates.begin(), candidates.begin() + CROP_CANDS_N);
}


std::ostream& operator<<(std::ostream& os, const Candidate& c)
{
	os << "Crop: "         << c.crop << ", S_compos: "   << c.S_compos;
	return os;
}

