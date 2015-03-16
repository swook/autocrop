#include "opencv2/opencv.hpp"

using namespace cv;

typedef unsigned int uint;

/**
 * Generates a pattern distinctiveness map
 *
 * 1) [Divide image into 9x9 patches]
 * 2) Perform PCA
 * 3) Project each patch into PCA space
 * 4) Take L1-norm and store to map
 */
Mat _getPatternDistinct(const Mat& img)
{
	uint H  = img.rows,
	     W  = img.cols,
	     Y  = H - 1,    // Limit of y indexing
	     X  = W - 1,
	     IH = H - 2,    // Inner width (sans 1-pixel border)
	     IW = W - 2,    // Inner height

	const uchar* row_p = img.ptr<uchar>(0); // Pixel values of i-1th row
	const uchar* row   = img.ptr<uchar>(1); // Pixel values of ith row
	const uchar* row_n;                     // Pixel values of i+1th row

	/******************************/
	/* Create list of 9x9 patches */
	/******************************/
	Mat patches     = Mat::zeros(IH*IW, 9, CV_8U); // List of patches
	uchar* patches_ = patches.ptr<uchar>(0);

	// Iterate over all inner pixels (patches)
	for (uint i = 0, y = 1; y < Y; y++) {
		row_n = img.ptr<uchar>(y + 1);
		for (uint x = 1; x < X; x++) {

			patches_[i]     = row_p[x - 1];
			patches_[i + 1] = row_p[x] ;
			patches_[i + 2] = row_p[x + 1];
			patches_[i + 3] = row  [x - 1];
			patches_[i + 4] = row  [x] ;
			patches_[i + 5] = row  [x + 1];
			patches_[i + 6] = row_n[x - 1];
			patches_[i + 7] = row_n[x] ;
			patches_[i + 8] = row_n[x + 1];
			i += 9;

		}
		row_p = row;
		row = row_n;
	}

	/*******/
	/* PCA */
	/*******/
	auto pca = PCA(patches, Mat(), CV_PCA_DATA_AS_ROW);

	Mat pca_pos  = Mat::zeros(IH * IW, 9, CV_32F); // Coordinates in PCA space
	Mat pca_norm = Mat::zeros(IH * IW, 1, CV_32F); // L1 norm of pca_pos

	pca.project(patches, pca_pos); 	                  // Project patches into PCA space
	reduce(abs(pca_pos), pca_norm, 1, CV_REDUCE_SUM); // Calc L1 norm

	// Pad with 1-pixel thick black border
	Mat out_inner = Mat(IH, IW, CV_32F, pca_norm.ptr<float>(0), 0);
	Mat out;
	copyMakeBorder(out_inner, out, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	return out;
}

/**
 * Generates a colour distinctiveness map
 *
 * 1)
 */
Mat _getColourDistinct(const Mat& img)
{
	auto out = Mat(img.size(), CV_32F);
	return out;
}

/**
 * Generates a saliency map using a method from Margolin et al. (2013)
 *
 * 1) Acquire pattern distinctiveness map
 * 2) Acquire colour distinctiveness map
 * 3) Calculate pixelwise multiplication of the two maps
 */
Mat getSaliency(const Mat& img)
{

	Mat patternD = _getPatternDistinct(img);
	//Mat colourD = _getColourDistinct(img);
	//return patternD.mul(colourD);
	return patternD;
}

/**
 * Crops a given image with a window of given aspect ratio (default: 1)
 * Uses a method from Chen et al. (2014)
 *
 * 1) Generate a saliency map using method from Margolin et al. (2013)
 */
Mat crop(const Mat& in, float w2hrat = 1.f)
{
	// Get grayscale image to work on
	auto grey = Mat(in.size(), in.type());
	cvtColor(in, grey, CV_BGR2GRAY);

	// Get saliency map
	Mat saliency = getSaliency(grey);
	return saliency;
}

