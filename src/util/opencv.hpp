#pragma once

#include "opencv2/opencv.hpp"

extern bool GRAPHICAL;

typedef unsigned int         uint;
typedef std::vector<cv::Mat> Mats;

/**
 * showImage shows a given image in a window with a given title
 */
void showImage(std::string title, const Mat& img);
void showImage(std::string title, const Mats& imgs);

/**
 * showImage shows a given image in a window with a given title and waits for
 * any user input
 */
void showImageAndWait(std::string title, const Mat& img);
void showImageAndWait(std::string title, const Mats& imgs);

/**
 * addGaussian adds a Gaussian of specified standard deviation centred at a
 * specified coordinate on a given image.
 */
void addGaussian(cv::Mat& img, uint x, uint y, float std, float weight);

