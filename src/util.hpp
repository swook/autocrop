#pragma once

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
void addGaussian(Mat& img, uint x, uint y, float std)
{
}
