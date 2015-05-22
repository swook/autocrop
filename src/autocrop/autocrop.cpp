#include <algorithm>

#include <opencv2/core.hpp>
using namespace cv;

#include "autocrop.hpp"
#include "../classify/Classifier.hpp"
#include "../features/feature.hpp"
#include "../saliency/saliency.hpp"
#include "../util/math.hpp"
#include "../util/opencv.hpp"

/**
 * Crops a given image with a window of given aspect ratio (default: 1)
 * Uses a method from Chen et al. (2014)
 *
 * 1) Generate a saliency map using method from Margolin et al. (2013)
 * 2)
 */
Mat autocrop(const Mat& in, float w2hrat)
{
	// Get saliency map and sum of values
	Mat saliency = getSaliency(in);
	float sum_saliency = sum(saliency)[0];

	// Calculate gradient map
	Mat gradient = getGradient(in);

	// Initialise classifier
	Classifier classifier;
	classifier.loadModel("Trained_model.yml");

	// Generate crop candidates
	const int MAX_CROP_CANDIDATES = 10000;
	std::vector<Candidate*> candidates;

#pragma omp parallel for
	for (int i = 0; i < MAX_CROP_CANDIDATES; i++)
	{
		// Generate single random crop
		Rect crop = randomCrop(in, w2hrat);
		Mat cr_saliency = saliency(crop);

		// Calculate content preservation
		float S_content = sum(cr_saliency)[0] / sum_saliency;
		if (S_content < 0.4) continue;

		// Calculate saliency composition
		float S_compos = classifier.classifyRaw(saliency, gradient, crop);

		// Calculate boundary simplicity
		int ch = crop.height,
		    cw = crop.width;
		Mat cr_gradient = gradient(crop);
		float S_boundary = .25f * (
			mean(cr_gradient(Rect(0,    0, cw, 1   )))[0] + // Top
			mean(cr_gradient(Rect(0, ch-1, cw, 1   )))[0] + // Bottom
			mean(cr_gradient(Rect(0,    1,  1, ch-2)))[0] + // Left
			mean(cr_gradient(Rect(cw-1, 1,  1, ch-2)))[0]   // Right
		);


		// Add to valid candidates list
#pragma omp critical
		candidates.push_back(new Candidate(crop, S_compos, S_boundary));
	}

	// Clean up
	classifier.clear();
	showImage("Saliency", saliency);
	saliency.release();
	gradient.release();

	const int    C = candidates.size();
	const float fC = (float) C;
	std::cout << C << " valid candidates left out of " << MAX_CROP_CANDIDATES << std::endl;

	// Sort by S_compos in ascending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_compos < j->S_compos;
		}
	);
#pragma omp parallel for
	for (int i = 0; i < candidates.size(); i++)
		candidates[i]->R_compos = (float) i / fC;


	// Sort by S_boundary in ascending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_boundary < j->S_boundary;
		}
	);
#pragma omp parallel for
	for (int i = 0; i < candidates.size(); i++)
		candidates[i]->R_boundary = (float) i / fC;


	// Calculate S_final
	const float w1 = 5.f,
	            w2 = 1.f;
#pragma omp parallel for
	for (int i = 0; i < candidates.size(); i++)
		candidates[i]->S_final = w1 * candidates[i]->R_compos +
		                         w2 * candidates[i]->R_boundary;


	// Sort by S_final in descending order
	std::sort(candidates.begin(), candidates.end(),
		[](Candidate* i, Candidate* j) {
			return i->S_final > j->S_final;
		}
	);
	Mat final_crop = in(candidates[0]->crop);


	// TMP
	std::cout << *candidates[0] << std::endl;
	std::cout << *candidates[1] << std::endl;
	std::cout << *candidates[2] << std::endl;
	std::cout << *candidates[3] << std::endl;
	std::cout << *candidates[4] << std::endl;
	std::cout << *candidates[5] << std::endl;
	showImage("Top candidates", {
		in(candidates[0]->crop),
		in(candidates[1]->crop),
		in(candidates[2]->crop),
		in(candidates[3]->crop),
		in(candidates[4]->crop)
	});

	// Free allocated structs
	for (int i = 0; i < candidates.size(); i++)
		free(candidates[i]);

	return final_crop;
}


std::ostream& operator<<(std::ostream& os, const Candidate& c)
{
	std::cout << "Crop: "         << c.crop
		  << ", S_compos: "   << c.S_compos
	          << ", S_boundary: " << c.S_boundary
	          << ", R_compos: "   << c.R_compos
	          << ", R_boundary: " << c.R_boundary
	          << ", S_final: "    << c.S_final
	;
}

