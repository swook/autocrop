#include <cmath>
#include <cstdarg>
#include <iostream>

#include "opencv2/core.hpp"
using namespace cv;

#include "math.hpp"

float median(std::vector<float>& v)
{
	const int n = v.size();
	assert(n > 0);
	if (n == 1) return v[0];

	std::sort(v.begin(), v.end());
	return v[n / 2];
}

float mean(std::vector<float>& v)
{
	const int n = v.size();
	if (n == 0) return 0.f;

	float sum = std::accumulate(std::begin(v), std::end(v), 0.0);
	return sum / n;
}

float var(std::vector<float>& v)
{
	const int n = v.size();
	if (n == 0) return 0.f;

	float _mean = mean(v);
	return sqrt(std::accumulate(std::begin(v), std::end(v), 0.f,
		[&](const float b, const float e) {
			float diff = e - _mean;
			return b + diff * diff;
		}) / n);
}


float randInt(const float min, const float max)
{
	return roundf(rand() / (float)RAND_MAX * (max - min) + min);
}


Rect randomCrop(const Mat& img, const float w2hrat)
{
	int h    = img.rows,
	    w    = img.cols,
	    minh = 8,
	    minw = 8;

	int x0, y0, dx, dy;
	while (1)
	{
		x0 = randInt(0,    w - minw);
		y0 = randInt(0,    h - minh);
		dy = randInt(minh, h - 1 - y0);
		if (w2hrat > 1e-5) // Valid aspect ratio
		{
			dx = roundf((float) dy * w2hrat);
			if (dx < minw || x0 + dx + 1 > w) continue;
		}
		else
			dx = randInt(minw, w - 1 - x0);

		return Rect(x0, y0, dx, dy);
	}
}


Rect randomCrop(const Mat& img, const Rect good_crop, const float thresh)
{
	Rect crop = randomCrop(img);
	while (cropOverlap(good_crop, crop) > thresh)
		crop = randomCrop(img);
	return crop;
}


float cropOverlap(const Rect crop1, const Rect crop2)
{
	int x1a = crop1.x,
	    y1a = crop1.y,
	    x1b = crop1.x + crop1.width,
	    y1b = crop1.y + crop1.height,
	    x2a = crop2.x,
	    y2a = crop2.y,
	    x2b = crop2.x + crop2.width,
	    y2b = crop2.y + crop2.height;

	int ow = max(0, min(x1b, x2b) - max(x1a, x2a)),
	    oh = max(0, min(y1b, y2b) - max(y1a, y2a));

	int oA = ow * oh;
	if (oA == 0) return 0;

	int A1 = crop1.width * crop1.height,
	    A2 = crop2.width * crop2.height;

	return oA / (float) (A1 + A2 - oA);
}

