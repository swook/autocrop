#pragma once

#include "opencv2/opencv.hpp"

extern bool GRAPHICAL;

typedef unsigned int         uint;
typedef std::vector<cv::Mat> Mats;

void showImage(const char* title, const cv::Mat& img);
void showImage(const char* title, const Mats&    imgs);

void showImageAndWait(const char* title, const cv::Mat& img);
void showImageAndWait(const char* title, const Mats&    imgs);

void addGaussian(cv::Mat& img, uint x, uint y, float std, float weight);
