#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

extern bool GRAPHICAL;

typedef unsigned int uint;

void showImage(const char* title, const Mat& img);

void showImageAndWait(const char* title, const Mat& img);

float var(std::vector<float>& v);

void addGaussian(Mat& img, uint x, uint y, float std, float weight);
