#include "opencv2/core.hpp"
using namespace cv;

#include "saliency.hpp"
#include "feature.hpp"


/**
 * getFeatureVector constructs a feature vector from a given image.
 * The expected input is a 3-channel BGR image.
 *
 * The steps are as follows:
 * 1) Get saliency map from given image
 * 2) Resize image to 4x4 to average saliency values
 * 3) Store 1/16ths as feature values
 * 4) Average 1/16 values to get feature values for 1/4ths
 * 5) Average 1/4 value to get feature value for whole image
 *
 * The final feature vector is:
 *   16x Average value of 1/16 patches
 *    4x Average value of 1/4  patches
 *    1x Average value of whole saliency map
 *   -----------
 *   21 features
 */
Mat getFeatureVector(const Mat& img)
{
	// Calculate saliency map
	Mat _saliency = getSaliency(img);

	// Resize to be 100x100
	Mat saliency;
	resize(_saliency, saliency, Size(4, 4), INTER_AREA);

	// Initialise feature vector
	Mat      feats = Mat(Size(1, FEATS_N), CV_64F);
	double* _feats = feats.ptr<double>(0);



	/* Visual Composition */

	// Add mean values for 1/16ths
	_feats[0]  = (double) saliency.at<float>(0, 0);
	_feats[1]  = (double) saliency.at<float>(0, 1);
	_feats[2]  = (double) saliency.at<float>(0, 2);
	_feats[3]  = (double) saliency.at<float>(0, 3);
	_feats[4]  = (double) saliency.at<float>(1, 0);
	_feats[5]  = (double) saliency.at<float>(1, 1);
	_feats[6]  = (double) saliency.at<float>(1, 2);
	_feats[7]  = (double) saliency.at<float>(1, 3);
	_feats[8]  = (double) saliency.at<float>(2, 0);
	_feats[9]  = (double) saliency.at<float>(2, 1);
	_feats[10] = (double) saliency.at<float>(2, 2);
	_feats[11] = (double) saliency.at<float>(2, 3);
	_feats[12] = (double) saliency.at<float>(3, 0);
	_feats[13] = (double) saliency.at<float>(3, 1);
	_feats[14] = (double) saliency.at<float>(3, 2);
	_feats[15] = (double) saliency.at<float>(3, 3);

	// Add mean values for 1/4ths
	_feats[16] = .25f * (_feats[0]  + _feats[1]  + _feats[4]  + _feats[5]);
	_feats[17] = .25f * (_feats[2]  + _feats[3]  + _feats[6]  + _feats[7]);
	_feats[18] = .25f * (_feats[8]  + _feats[9]  + _feats[12] + _feats[13]);
	_feats[19] = .25f * (_feats[10] + _feats[11] + _feats[14] + _feats[15]);

	// Add mean value for all pixels in saliency map
	_feats[20] = .25f * (_feats[16] + _feats[17] + _feats[18] + _feats[19]);



	/* Boundary Simplicity */
	// TODO

	// Blur to remove high frequency textures

	// Calculate gradient of image

	// Take average gradient along boundary



	/* Content Preservation */
	// TODO

	// Ratio of saliency energy inside crop to total saliency energy


	// Return full feature vector
	return feats;
}
