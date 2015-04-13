#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
using namespace cv;

#include "saliency.hpp"
#include "train.hpp"

void Trainer::add(Mat& img)
{
}

void Trainer::train()
{
}


/**
 * getFeatureVector constructs a feature vector from a given image.
 * The expected input is a 3-channel BGR image.
 *
 * The steps are as follows:
 * 1) Get saliency map from given image
 * 2) Resize image to 100x100
 * 3) Divide into 1/16ths and add average as feature values
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
Mat getFeatureVector(Mat& img)
{
	// Calculate saliency map
	Mat _saliency = getSaliency(img);

	Mat saliency;
	resize(_saliency, saliency, Size(100, 100));

	// Initialise feature vector
	const int featsN = 21;
	double feats[featsN];

	// Add mean values for 1/16ths
	feats[0]  = mean(saliency(Rect(0,   0, 25, 25)))[0],
	feats[1]  = mean(saliency(Rect(25,  0, 25, 25)))[0],
	feats[2]  = mean(saliency(Rect(50,  0, 25, 25)))[0],
	feats[3]  = mean(saliency(Rect(75,  0, 25, 25)))[0],
	feats[4]  = mean(saliency(Rect(0,  25, 25, 25)))[0],
	feats[5]  = mean(saliency(Rect(25, 25, 25, 25)))[0],
	feats[6]  = mean(saliency(Rect(50, 25, 25, 25)))[0],
	feats[7]  = mean(saliency(Rect(75, 25, 25, 25)))[0],
	feats[8]  = mean(saliency(Rect(0,  50, 25, 25)))[0],
	feats[9]  = mean(saliency(Rect(25, 50, 25, 25)))[0],
	feats[10] = mean(saliency(Rect(50, 50, 25, 25)))[0],
	feats[11] = mean(saliency(Rect(75, 50, 25, 25)))[0],
	feats[12] = mean(saliency(Rect(0,  75, 25, 25)))[0],
	feats[13] = mean(saliency(Rect(25, 75, 25, 25)))[0],
	feats[14] = mean(saliency(Rect(50, 75, 25, 25)))[0],
	feats[15] = mean(saliency(Rect(75, 75, 25, 25)))[0];

	// Add mean values for 1/4ths
	feats[16] = .25f * (feats[0]  + feats[1]  + feats[4]  + feats[5]);
	feats[17] = .25f * (feats[2]  + feats[3]  + feats[6]  + feats[7]);
	feats[18] = .25f * (feats[8]  + feats[9]  + feats[12] + feats[13]);
	feats[19] = .25f * (feats[10] + feats[11] + feats[14] + feats[15]);

	// Add mean value for all pixels in saliency map
	feats[20] = .25f * (feats[16] + feats[17] + feats[18] + feats[19]);

	// Return constructed vector
	return Mat(Size(1, featsN), CV_32F, feats);
}
