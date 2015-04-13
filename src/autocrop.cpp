#include "opencv2/core.hpp"
using namespace cv;

#include "saliency.hpp"

/**
 * Crops a given image with a window of given aspect ratio (default: 1)
 * Uses a method from Chen et al. (2014)
 *
 * 1) Generate a saliency map using method from Margolin et al. (2013)
 * 2)
 */
Mat crop(const Mat& in, float w2hrat = 1.f)
{
	// Get saliency map
	Mat saliency = getSaliency(in);
	return saliency;
}

