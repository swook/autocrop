#pragma once

#include "opencv2/opencv.hpp"

using namespace cv;

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

float var(std::vector<float>& v) {
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
