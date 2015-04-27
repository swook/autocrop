#include "opencv2/core.hpp"
using namespace cv;

#include "SLIC.hpp"
#include "../util/opencv.hpp"
#include "../util/math.hpp"

extern "C" {
#include "vl/generic.h"
#include "vl/slic.h"
}


/**
 * Calculates SLIC segmentation for a given LAB image
 */
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
	vl_size regionSize    = 50,
		minRegionSize = 35;
	printf("\nSLIC parameters:\n- regionSize: %llu\n- minRegionSize: %llu\n",
	       regionSize, minRegionSize);

	vl_slic_segment(segmentation.data(), img_vl, W, H, img.channels(),
			regionSize, 800, minRegionSize);

	//return; // Skip visualisation. Comment out to tune parameters.

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
	//char* title[20];
	//sprintf(title, "SLIC: %d", rand());
	//showImage(title, vis);
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
