#include <stdexcept>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

#include "../util/opencv.hpp"
#include "../saliency/saliency.hpp"
#include "feature.hpp"


/**
 * Convenience method for providing source image, not saliency map and edge map.
 *
 * This version is used for cases where only one data point exists per image
 * while datasets should use the other method to reduce calculation of saliency
 * and edge maps.
 */
Mat getFeatureVector(const Mat& img, const Rect crop)
{
	// Calculate saliency map
	Mat _saliency = getSaliency(img);

	// Calculate gradient image
	Mat _grad = getGrad(img);

	// Calculate feature vector
	Mat feats = getFeatureVector(_saliency, _grad, crop);

	return feats;
}


/**
 * getFeatureVector constructs a feature vector from a given saliency map and
 * gradient map.
 *
 * The final feature vector length is:
 *   21 Visual composition
 *    1 Boundary simplicity
 *    1 Content preservation
 *   -----------
 *   23 features
 */
cv::Mat getFeatureVector(const cv::Mat& saliency, const cv::Mat& grad,
	const cv::Rect crop)
{
	int h = saliency.rows,
	    w = saliency.cols;

	Mat cr_saliency = saliency(crop);
	Mat cr_grad     = grad(crop);

	showImage("cropped saliency", cr_saliency);
	showImageAndWait("cropped gradient", cr_grad);


	// Resize cropped saliency map to be 4x4. Use INTER_AREA to average pixel
	// values
	Mat _saliency;
	resize(cr_saliency, _saliency, Size(4, 4), INTER_AREA);

	// Initialise feature vector
	Mat     feats = Mat(Size(FEATS_N, 1), CV_32F);
	float* _feats = feats.ptr<float>(0);


	/**
	 * Visual Composition
	 * - 16 + 4 + 1 = 21 features
	 *
	 * 1) Resize image to 4x4 to average saliency values
	 * 2) Store 1/16ths as feature values
	 * 3) Average 1/16 values to get feature values for 1/4ths
	 * 4) Average 1/4 value to get feature value for whole image
	 */

	// Add mean values for 1/16ths
	_feats[0]  = _saliency.at<float>(0, 0);
	_feats[1]  = _saliency.at<float>(0, 1);
	_feats[2]  = _saliency.at<float>(0, 2);
	_feats[3]  = _saliency.at<float>(0, 3);
	_feats[4]  = _saliency.at<float>(1, 0);
	_feats[5]  = _saliency.at<float>(1, 1);
	_feats[6]  = _saliency.at<float>(1, 2);
	_feats[7]  = _saliency.at<float>(1, 3);
	_feats[8]  = _saliency.at<float>(2, 0);
	_feats[9]  = _saliency.at<float>(2, 1);
	_feats[10] = _saliency.at<float>(2, 2);
	_feats[11] = _saliency.at<float>(2, 3);
	_feats[12] = _saliency.at<float>(3, 0);
	_feats[13] = _saliency.at<float>(3, 1);
	_feats[14] = _saliency.at<float>(3, 2);
	_feats[15] = _saliency.at<float>(3, 3);

	// Add mean values for 1/4ths
	_feats[16] = .25f * (_feats[0]  + _feats[1]  + _feats[4]  + _feats[5]);
	_feats[17] = .25f * (_feats[2]  + _feats[3]  + _feats[6]  + _feats[7]);
	_feats[18] = .25f * (_feats[8]  + _feats[9]  + _feats[12] + _feats[13]);
	_feats[19] = .25f * (_feats[10] + _feats[11] + _feats[14] + _feats[15]);

	// Add mean value for all pixels in saliency map
	_feats[20] = .25f * (_feats[16] + _feats[17] + _feats[18] + _feats[19]);



	/**
	 * Boundary Simplicity
	 * - 1 feature
	 */
	// Take average gradient along boundary
	_feats[21] = .25f * (
			mean(grad(Rect(0, 0,   w, 1)))[0]   + // Top
			mean(grad(Rect(0, h-1, w, 1)))[0]   + // Bottom
			mean(grad(Rect(0,   1, 1, h-2)))[0] + // Left
			mean(grad(Rect(w-1, 1, 1, h-2)))[0]   // Right
	             );


	/**
	 * Content Preservation
	 * - 1 feature
	 */
	// Ratio of saliency energy inside crop to total saliency energy
	_feats[22] = sum(cr_saliency)[0] / sum(saliency)[0];


	// Return full feature vector
	return feats;
}


// Get gradient
Mat getGrad(const Mat& img)
{
	// Blur to remove high frequency textures
	Mat blurred;
	auto kernel_size = Size(3, 3);
	blur(img, blurred, kernel_size);

	// Calculate gradient of image
	Mat out;
	Sobel(blurred, out, CV_32F, 1, 1);

	return out;
}

