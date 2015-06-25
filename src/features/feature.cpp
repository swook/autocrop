#include <stdexcept>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

#include "feature.hpp"
#include "FeatMat.hpp"
#include "../saliency/saliency.hpp"
#include "../util/opencv.hpp"

Mat getFeatureVector(const Mat& img)
{
	// Calculate saliency map
	Mat saliency = getSaliency(img);

	// Calculate gradient image
	Mat gradient = getGradient(img);

	// Calculate feature vector
	return getFeatureVector(saliency, gradient);
}

Mat getFeatureVector(const Mat& img, const Rect crop)
{
	// Calculate saliency map
	Mat saliency = getSaliency(img);

	// Calculate gradient image
	Mat gradient = getGradient(img);

	return getFeatureVector(saliency, gradient, crop);
}

Mat getFeatureVector(const Mat& saliency, const Mat& gradient,
	const Rect crop)
{
	return getFeatureVector(saliency(crop), gradient(crop));
}

Mat getFeatureVector(const Mat& saliency, const Mat& gradient)
{
	int h  = saliency.rows,
	    w  = saliency.cols;

	// Resize saliency map to be 8x8. Use INTER_AREA to average pixel values
	Mat _saliency;
#if spsm3
	resize(saliency, _saliency, Size(8, 8), INTER_AREA);
#else
	resize(saliency, _saliency, Size(4, 4), INTER_AREA);
#endif

	// Initialise feature vector
	Mat     feats = Mat(Size(FEATS_N, 1), CV_32F);
	float* _feats = feats.ptr<float>(0);

	int i = 0; // Index in feature vector

	// Add mean values for 1/64ths
	float* p_saliency = _saliency.ptr<float>(0);
#if spsm3
	for (int c = 0; c < 64; c++)
	{
		_feats[i] = p_saliency[i];
		i++;
	}
#endif

	// Add mean values for 1/16ths
#if spsm3
	for (int j = 0; j < 4; j++)
		for (int k = 0; k < 4; k++)
		{
			_feats[i] = .25f * (
				_feats[16 * j + 2 * k]     +
				_feats[16 * j + 2 * k + 1] +
				_feats[16 * j + 2 * k + 8] +
				_feats[16 * j + 2 * k + 9]
			);
			i++;
		}
#else
	for (int c = 0; c < 16; c++)
	{
		_feats[i] = p_saliency[i];
		i++;
	}
#endif

	// Add mean values for 1/4ths
	for (int j = 0; j < 2; j++)
		for (int k = 0; k < 2; k++)
		{
			_feats[i] = 0.25f * (
#if spsm3
				_feats[64 + 8 * j + 2 * k]     +
				_feats[64 + 8 * j + 2 * k + 1] +
				_feats[64 + 8 * j + 2 * k + 4] +
				_feats[64 + 8 * j + 2 * k + 5]
#else
				_feats[8 * j + 2 * k]     +
				_feats[8 * j + 2 * k + 1] +
				_feats[8 * j + 2 * k + 4] +
				_feats[8 * j + 2 * k + 5]
#endif
			);
			i++;
		}

	// Add mean value for all pixels in saliency map
	_feats[i] = .25f * (
		_feats[i - 4] +
		_feats[i - 3] +
		_feats[i - 2] +
		_feats[i - 1]
	);
	i++;


	// Boundary simplicity features
	int b = 2; // No. of breadth of "boundary"
	_feats[i] = mean(gradient(Rect(0,     0,     w, b)))[0]; i++; // Top
	_feats[i] = mean(gradient(Rect(0,     h-1-b, w, b)))[0]; i++; // Bottom
	_feats[i] = mean(gradient(Rect(0,     0,     b, h)))[0]; i++; // Left
	_feats[i] = mean(gradient(Rect(w-1-b, 0,     b, h)))[0]; i++; // Right


	// Add sum of all pixels in saliency map
	_feats[i] = sum(_saliency)[0];


	return feats;
}


Mat getGradient(const Mat& img)
{
	// Set to single-channel
	Mat gray;
	if (img.channels() == 3)
	{
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else gray = img;

	// Calculate gradient of image
	Mat grad;
	Sobel(gray, grad, CV_32F, 1, 1, 3);
	grad = abs(grad);

	return abs(grad);
}

