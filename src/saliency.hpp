#pragma once

#include "opencv2/opencv.hpp"
using namespace cv;

#include "SLIC.hpp"
#include "util.hpp"

/**
 * Generates a pattern distinctiveness map
 *
 * 1) [Divide image into 9x9 patches]
 * 2) Perform PCA
 * 3) Project each patch into PCA space
 * 4) Take L1-norm and store to map
 */
Mat _getPatternDistinct(const Mat& img, std::vector<vl_uint32>& segmentation,
                        std::vector<float>& spxl_vars, float var_thresh)
{
	uint H  = img.rows,
	     W  = img.cols,
	     Y  = H - 1,    // Limit of y indexing
	     X  = W - 1,
	     IH = H - 2,    // Inner width (sans 1-pixel border)
	     IW = W - 2;    // Inner height

	const uchar* row_p = img.ptr<uchar>(0); // Pixel values of i-1th row
	const uchar* row   = img.ptr<uchar>(1); // Pixel values of ith row
	const uchar* row_n;                     // Pixel values of i+1th row

	/******************************/
	/* Create list of 9x9 patches */
	/******************************/
	auto _patches = std::vector<uchar>(0);
	_patches.reserve(X * Y * 9);

	// Patches in superpixels with var above var_thresh
	auto _distpatches = std::vector<uchar>(0);
	_distpatches.reserve(X * Y * 9);

	// Iterate over all inner pixels (patches) with variance above threshold
	uint i = 0, spxl_i;
	for (uint y = 1; y < Y; y++) {
		row_n = img.ptr<uchar>(y + 1);
		for (uint x = 1; x < X; x++) {
			spxl_i = segmentation[y*X + x];
			if (spxl_vars[spxl_i] > var_thresh) {
				_distpatches.push_back(row_p[x - 1]);
				_distpatches.push_back(row_p[x]    );
				_distpatches.push_back(row_p[x + 1]);
				_distpatches.push_back(row  [x - 1]);
				_distpatches.push_back(row  [x]    );
				_distpatches.push_back(row  [x + 1]);
				_distpatches.push_back(row_n[x - 1]);
				_distpatches.push_back(row_n[x]    );
				_distpatches.push_back(row_n[x + 1]);
				i++;
			}

			_patches.push_back(row_p[x - 1]);
			_patches.push_back(row_p[x]    );
			_patches.push_back(row_p[x + 1]);
			_patches.push_back(row  [x - 1]);
			_patches.push_back(row  [x]    );
			_patches.push_back(row  [x + 1]);
			_patches.push_back(row_n[x - 1]);
			_patches.push_back(row_n[x]    );
			_patches.push_back(row_n[x + 1]);

		}
		row_p = row;
		row = row_n;
	}
	_distpatches.shrink_to_fit();
	auto distpatches = Mat(i, 9, CV_8U, _distpatches.data());
	auto patches     = Mat(X*Y, 9, CV_8U, _patches.data());

	/*******/
	/* PCA */
	/*******/
	auto pca = PCA(distpatches, Mat(), CV_PCA_DATA_AS_ROW);

	Mat pca_pos  = Mat::zeros(IH * IW, 9, CV_32F); // Coordinates in PCA space
	Mat pca_norm = Mat::zeros(IH * IW, 1, CV_32F); // L1 norm of pca_pos

	pca.project(patches, pca_pos); 	                  // Project patches into PCA space
	reduce(abs(pca_pos), pca_norm, 1, CV_REDUCE_SUM); // Calc L1 norm

	// Pad with 1-pixel thick black border
	Mat out_inner = Mat(IH, IW, CV_32F, pca_norm.ptr<float>(0), 0);
	Mat out;
	copyMakeBorder(out_inner, out, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	// Normalise
	Mat out_norm;
	normalize(out, out_norm, 0.f, 1.f, NORM_MINMAX);
	return out_norm;
}

/**
 * Generates a colour distinctiveness map
 *
 * 1) Calculate average colour per SLIC region
 * 2) Calculate sum of euclidean distance between colours
 */
Mat _getColourDistinct(const Mat& img, std::vector<vl_uint32>& segmentation,
                       uint spxl_n)
{
	uint H  = img.rows,
	     W  = img.cols,
	     HW = H * W;

	// 1. Aggregate colours of regions
	auto spxl_cols = std::vector<Vec3f>(spxl_n);
	auto spxl_cnts = std::vector<uint>(spxl_n);
	// Allocate
	for (uint i = 0; i < spxl_n; i++)
		spxl_cols[i] = Vec3f();

	// Aggregate Lab colour values
	Vec3f col;
	for (uint idx = 0, j = 0; j < H; j++)
		for (uint i = 0; i < W; i++) {
			idx = segmentation[j*W + i];
			col = spxl_cols[idx];
			col[0] += (float)img.ptr<Vec3b>(j)[i][0];
			col[1] += (float)img.ptr<Vec3b>(j)[i][1];
			col[2] += (float)img.ptr<Vec3b>(j)[i][2];
			spxl_cols[idx] = col;
			spxl_cnts[idx]++;
		}

	// Divide by no. of pixels
	for (uint i = 0; i < spxl_n; i++) {
		spxl_cols[i][0] /= spxl_cnts[i];
		spxl_cols[i][1] /= spxl_cnts[i];
		spxl_cols[i][2] /= spxl_cnts[i];
	}

	// 2. Aggregate colour distances
	auto spxl_dist = std::vector<float>(spxl_n);
	float dist;
	for (uint i1 = 0; i1 < spxl_n; i1++) {
		dist = 0.f;
		for (uint i2 = 0; i2 < spxl_n; i2++) {
			if (i1 == i2) continue;
			dist += norm(spxl_cols[i1] - spxl_cols[i2]);
		}
		spxl_dist[i1] = dist / spxl_n;
	}

	// 3. Assign distance value to output colour distinctiveness map
	auto out = Mat(img.size(), CV_32F);
	for (uint idx = 0, j = 0; j < H; j++)
		for (uint i = 0; i < W; i++) {
			idx = segmentation[j*W + i];
			out.ptr<float>(j)[i] = spxl_dist[idx];
		}

	// Normalise
	Mat out_norm;
	normalize(out, out_norm, 0.f, 1.f, NORM_MINMAX);
	return out_norm;
}

/**
 * Generates a Gaussian weight map
 *
 * 1) Threshold given distinctiveness map with thresholds in 0:0.1:1
 * 2) Compute centre of mass
 * 3) Place Gaussian with standard deviation 1000 at CoM
 *    (Weight according to threshold)
 */
Mat _getWeightMap(Mat& D)
{
	auto out = Mat(D.size(), CV_32F);

	// Normalise
	Mat out_norm;
	normalize(out, out_norm, 0.f, 1.f, NORM_MINMAX);
	return out_norm;
}

/**
 * Generates a saliency map using a method from Margolin et al. (2013)
 *
 * 1) Acquire pattern distinctiveness map
 * 2) Acquire colour distinctiveness map
 * 3) Calculate pixelwise multiplication of the two maps
 */
const float maxSize = 600.f;

Mat getSaliency(const Mat& img)
{
	Mat img_BGR;

	uint H  = img.rows,
	     W  = img.cols,
	     HW = H * W;

	// Scale image to have not more than maxSize pixels on its larger
	// dimension
	float scale = (float) max(H, W) / maxSize;
	if (scale > 1.f) {
		resize(img, img_BGR, Size(W / scale, H / scale));

		W = W / scale;
		H = H / scale;
	} else {
		img_BGR = img;
	}

	// Get grayscale image to work on
	auto img_grey = Mat(img_BGR.size(), img_BGR.type());
	cvtColor(img_BGR, img_grey, CV_BGR2GRAY);

	auto img_lab = Mat(img_BGR.size(), img_BGR.type());
	cvtColor(img_BGR, img_lab, CV_BGR2Lab);

	// Get SLIC superpixels
	auto segmentation = std::vector<vl_uint32>(H*W);
	_getSLICSegments(img_lab, segmentation);

	// Calculate variance of super pixels
	auto spxl_n = std::accumulate(segmentation.begin(),
		segmentation.end(), 0, [&](vl_uint32 b, vl_uint32 n) {
			return n > b ? n : b;
		}) + 1;
	printf("Calculated %d superpixels.\n", spxl_n);
	auto spxl_vars  = std::vector<float>(spxl_n);
	auto var_thresh = _getSLICVariances(img_grey, segmentation, spxl_vars);

	// Compute distinctiveness maps
	Mat patternD = _getPatternDistinct(img_grey, segmentation, spxl_vars, var_thresh);
	showImage("Pattern Distinctiveness", patternD);

	Mat colourD = _getColourDistinct(img_lab, segmentation, spxl_n);
	showImage("Colour Distinctiveness", colourD);

	Mat D = patternD.mul(colourD);
	showImage("Distinctiveness", D);

	Mat G = _getWeightMap(D);
	showImage("Gaussian Weight Map", G);

	Mat out;
	normalize(D.mul(G), out, 0.f, 1.f, NORM_MINMAX);
	showImage("Saliency Map", out);

	// Scale back to original size for further processing
	if (scale > 1.f) {
		Mat out_scaled = Mat();
		resize(out, out_scaled, img.size());
		std::swap(out, out_scaled);
	}
	return out;
}
