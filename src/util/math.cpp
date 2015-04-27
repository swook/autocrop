#include <cmath>
#include <cstdarg>

#include "opencv2/core.hpp"
using namespace cv;

#include "math.hpp"

/**
 * Calculates the variance of values in a given list of floats
 */
float var(std::vector<float>& v)
{
	auto n = v.size();
	if (n == 0) return 0.f;

	auto sum = std::accumulate(std::begin(v), std::end(v), 0.0);
	auto mean = sum / n;
	return sqrt(std::accumulate(std::begin(v), std::end(v), 0.f,
		[&](const float b, const float e) {
			float diff = e - mean;
			return b + diff * diff;
		}) / n);
}


float randInt(const float min, const float max)
{
	return roundf(rand() / (float)RAND_MAX * (max - min) + min);
}


/**
 * Returns a random crop window for a given image.
 *
 * This is used to generate "bad" crops for the trainer.
 *
 * TODO: Ensure overlap with "good" crop is not too big
 */
Rect randomCrop(const Mat& img)
{
	int h = img.rows,
	    w = img.cols;

	int x0 = randInt(0, w - 2),
	    y0 = randInt(0, h - 2),
	    dx = randInt(1, w - 1 - x0),
	    dy = randInt(1, h - 1 - y0);

	return Rect(x0, y0, dx, dy);
}

