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
	resize(saliency, _saliency, Size(8, 8), INTER_AREA);

	// Initialise feature vector
	Mat     feats = Mat(Size(FEATS_N, 1), CV_32F);
	float* _feats = feats.ptr<float>(0);

	int i = 0; // Index in feature vector

	// Add mean values for 1/64ths
	float* p_saliency = _saliency.ptr<float>(0);
	for (int c = 0; c < 64; c++)
	{
		_feats[i] = p_saliency[i];
		i++;
	}

	// Add mean values for 1/16ths
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

	// Add mean values for 1/4ths
	for (int j = 0; j < 2; j++)
		for (int k = 0; k < 2; k++)
		{
			_feats[i] = 0.25f * (
				_feats[64 + 8 * j + 2 * k]     +
				_feats[64 + 8 * j + 2 * k + 1] +
				_feats[64 + 8 * j + 2 * k + 4] +
				_feats[64 + 8 * j + 2 * k + 5]
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

	return feats;
}


Mat getGradient(const Mat& img)
{
	// Set to single-channel
	Mat gray = img;
	if (img.channels() == 3)
	{
		cvtColor(img, gray, CV_BGR2GRAY);
	}

	// Blur to remove high frequency textures
	Mat   blurred;
	Size  kernel_size = Size(3, 3);
	float sigma       = 1.f;
	GaussianBlur(gray, blurred, kernel_size, sigma);

	// Calculate gradient of image
	Mat grad;
	Sobel(blurred, grad, CV_32F, 1, 1, 3);

	// Fix range
	Mat out;
	normalize(grad, out, 0.f, 1.f, NORM_MINMAX);

	return out;
}

