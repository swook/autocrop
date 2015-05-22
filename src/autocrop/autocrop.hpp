#pragma once

#include <opencv2/core.hpp>

#include "../saliency/saliency.hpp"

struct Candidate
{
	const cv::Rect crop;
	const float S_compos;
	const float S_boundary;
	float R_compos;
	float R_boundary;
	float S_final;

	Candidate(const cv::Rect crop,
	      const float S_compos,
	      const float S_boundary)
	: crop(crop), S_compos(S_compos), S_boundary(S_boundary) {}
};

Mat autocrop(const Mat& in, float w2hrat = 1.f);
std::ostream& operator<<(std::ostream& os, const Candidate& c);

