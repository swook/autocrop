#include <cmath>
#include <cstdarg>

#include "opencv2/core.hpp"
using namespace cv;

#include "math.hpp"

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


Rect randomCrop(const Mat& img)
{
	int h    = img.rows,
	    w    = img.cols,
	    minw = 4,
	    minh = 4;

	int x0 = randInt(0,    w - minw),
	    y0 = randInt(0,    h - minh),
	    dx = randInt(minw, w - 1 - x0),
	    dy = randInt(minh, h - 1 - y0);

	return Rect(x0, y0, dx, dy);
}

