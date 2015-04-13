#pragma once

#include "opencv2/opencv.hpp"

extern bool GRAPHICAL;

typedef unsigned int uint;

void showImage(const char* title, const cv::Mat& img);

void showImageAndWait(const char* title, const cv::Mat& img);

float var(std::vector<float>& v);

void addGaussian(cv::Mat& img, uint x, uint y, float std, float weight);
