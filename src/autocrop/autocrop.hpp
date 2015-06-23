#pragma once

#include <opencv2/core.hpp>

#include "../saliency/saliency.hpp"
#include "../classify/Classifier.hpp"

struct Candidate
{
	const cv::Rect crop;
	float S_compos;

	Candidate(const cv::Rect crop,
	          const float S_compos)
	: crop(crop), S_compos(S_compos) {}
};
typedef std::vector<Candidate*> Candidates;

std::ostream& operator<<(std::ostream& os, const Candidate& c);

/**
 * Crops a given image with a window of given aspect ratio (default: any)
 * Uses a method from Chen et al. (2014)
 */
Mat autocrop(const Mat& in, float w2hrat = 0.f);

Rect getBestCrop(const Mat& in, float w2hrat = 0.f);
Rect getBestCrop(const Mat& saliency, const Mat& gradient, float w2hrat = 0.f);

#define CROP_CANDS_N 10

Candidates getCropCandidates(const Classifier& classifier, const Mat& saliency,
	const Mat& gradient, float w2hrat = 0.f);

