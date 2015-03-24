#pragma once

#include "opencv2/opencv.hpp"

extern "C" {
#include "vl/generic.h"
#include "vl/slic.h"
}

#include "util.hpp"

using namespace cv;

typedef unsigned int uint;

void _getSLICSegments(const Mat& img, std::vector<vl_uint32>& segmentation)
{
	uint H = img.rows,
	     W = img.cols,
	     HW = H * W;

	// Convert format from LABLAB to LLAABB (for vlfeat)
	auto img_vl = new float[HW * 3];
	auto img_   = img.ptr<Vec3b>(0);
	for (uint j = 0; j < H; j++) {
		for (uint i = 0; i < W; i++) {
			img_vl[j * W + i]          = img_[j * W + i][0];
			img_vl[j * W + i + HW]     = img_[j * W + i][1];
			img_vl[j * W + i + HW * 2] = img_[j * W + i][2];
		}
	}

	// Run SLIC code from vlfeat
	vl_size regionSize    = 30,
		minRegionSize = 5;
	printf("\nSLIC parameters:\n- regionSize: %llu\n- minRegionSize: %llu\n",
	       regionSize, minRegionSize);

	vl_slic_segment(segmentation.data(), img_vl, W, H, img.channels(),
			regionSize, 1000, minRegionSize);

	// Visualise segmentation
	Mat vis;
	cvtColor(img, vis, CV_Lab2BGR);
	int** labels = new int*[H];
	for (uint j = 0; j < H; j++) {
		labels[j] = new int[W];
		for (uint i = 0; i < W; i++)
			labels[j][i] = (int) segmentation[j*W + i];
	}

	int label, labelTop, labelBottom, labelLeft, labelRight;
	for (uint j = 1; j < H - 1; j++) {
		for (uint i = 1; i < W - 1; i++) {
			label       = labels[j][i];
			labelTop    = labels[j - 1][i];
			labelBottom = labels[j + 1][i];
			labelLeft   = labels[j][i - 1];
			labelRight  = labels[j][i + 1];
			if (label != labelTop  || label != labelBottom ||
			    label != labelLeft || label != labelRight) {
				vis.at<Vec3b>(j, i)[0] = 0;
				vis.at<Vec3b>(j, i)[1] = 0;
				vis.at<Vec3b>(j, i)[2] = 255;
			}
		}
	}
	showImage("SLIC", vis);
}

float _getSLICVariances(Mat& grey, std::vector<vl_uint32>& segmentation,
                        std::vector<float>& vars)
{
	uint n  = vars.size(),
	     HW = grey.cols * grey.rows;

	// 1. Aggregate pixels by super pixel
	auto spxl_vals = std::vector<std::vector<float>>(n);
	for (uint i = 0; i < n; i++) {
		spxl_vals[i] = std::vector<float>(0);
		spxl_vals[i].reserve(20);
	}

	vl_uint32 spxl_id;
	float     spxl_val;
	for (uint i = 0; i < HW; i++) {
		spxl_id = segmentation[i];
		spxl_val = (float) grey.ptr<uchar>(0)[i];
		spxl_vals[spxl_id].push_back(spxl_val);
	}

	// 2. Calculate variance of group of pixels
	for (uint i = 0; i < n; i++)
		vars[i] = var(spxl_vals[i]);

	// 3. Calculate variance threshold (25% with highest variance)
	auto vars_sorted = vars;
	std::sort(vars_sorted.begin(), vars_sorted.end());
	return vars_sorted[n - n / 4];
}

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
 * 1)
 */
Mat _getColourDistinct(const Mat& img, std::vector<vl_uint32>& segmentation,
                       uint spxl_n)
{
	// 1. Aggregate pixels by super pixel
	auto spxl_vals = std::vector<std::vector<float>>(spxl_n);
	for (uint i = 0; i < spxl_n; i++) {
		spxl_vals[i] = std::vector<float>(0);
		spxl_vals[i].reserve(20);
	}

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

	Mat colourD  = _getColourDistinct(img_lab, segmentation, spxl_n);
	showImage("Colour Distinctiveness", colourD);

	Mat out;
	normalize(patternD.mul(colourD), out, 0.f, 1.f, NORM_MINMAX);
	showImage("Saliency Map", out);

	// Scale back to original size for further processing
	if (scale > 1.f) {
		Mat out_scaled = Mat();
		resize(out, out_scaled, img.size());
		std::swap(out, out_scaled);
	}
	return out;
}
