#pragma once

#include <cmath>

#include "opencv2/opencv.hpp"
using namespace cv;

typedef unsigned int uint;

void showImage(const char* title, const Mat& img)
{
	std::cout << "\nShowing image: \"" << title << "\"." << std::endl;
	namedWindow(title, CV_WINDOW_NORMAL);
	imshow(title, img);
}

void showImageAndWait(const char* title, const Mat& img)
{
	showImage(title, img);
	std::cout << "Press any key to continue..." << std::endl;
	waitKey(0);
}

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

/**
 * Adds a Gaussian of specified standard deviation centred at a specified
 * coordinate on a given image
 */
void addGaussian(Mat& img, uint x, uint y, float std, float weight)
{
	uint H  = img.rows,
	     W  = img.cols,
	     HW = H * W;

	const float a =  1.f / (std * std * 2.f * M_PI),
	            b = -.5f / (std * std);

	auto  done = std::vector<bool>(HW);  // Mark if calculation done
	float dy2, dx2, g;
	int   dy, dx, j, i, j_, i_; // Mirrored pixels to check
	uint  ji_;

	for (j = 0; j < H; j++)
	{
		dy  = abs(j - (int)y);
		dy2 = dy * dy;
		for (i = 0; i < W; i++)
		{
			if (done[j * W + i]) continue;

			dx  = abs(i - (int)x);
			dx2 = dx * dx;
			g   = weight * a * exp(b * (dx2 + dy2));

			j_ = y + dy; i_ = x + dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y - dy; i_ = x + dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y + dy; i_ = x - dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y - dy; i_ = x - dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}
		}
	}
}
