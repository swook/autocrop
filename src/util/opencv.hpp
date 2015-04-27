#pragma once

#include "opencv2/opencv.hpp"

extern bool GRAPHICAL;

typedef unsigned int         uint;
typedef std::vector<cv::Mat> Mats;

void showImage(std::string title, const Mat& img);
void showImage(std::string title, const Mats& imgs);

void showImageAndWait(std::string title, const Mat& img);
void showImageAndWait(std::string title, const Mats& imgs);

void addGaussian(cv::Mat& img, uint x, uint y, float std, float weight);

