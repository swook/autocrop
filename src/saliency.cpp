#include "../lib/BMS/BMS.cpp"

#include "SLIC.hpp"
#include "util.hpp"

const float _9 = 1.f / 9.f;

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
	float p1, p2, p3, p4, p5, p6, p7, p8, p9, m;
	for (uint y = 1; y < Y; y++) {
		row_n = img.ptr<uchar>(y + 1);
		for (uint x = 1; x < X; x++) {
			p1 = row_p[x - 1]; p2 = row_p[x]; p3 = row_p[x + 1];
			p4 =   row[x - 1]; p5 =   row[x]; p6 =   row[x + 1];
			p7 = row_n[x - 1]; p8 = row_n[x]; p9 = row_n[x + 1];
			m = _9 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9);

			spxl_i = segmentation[y*X + x];
			// TODO: "To disregard lighting effects we a-priori
			//        subtract from each patch its mean value."
			if (spxl_vars[spxl_i] > var_thresh)
			{
				_distpatches.push_back(p1 - m);
				_distpatches.push_back(p2 - m);
				_distpatches.push_back(p3 - m);
				_distpatches.push_back(p4 - m);
				_distpatches.push_back(p5 - m);
				_distpatches.push_back(p6 - m);
				_distpatches.push_back(p7 - m);
				_distpatches.push_back(p8 - m);
				_distpatches.push_back(p9 - m);
				i++;
			}

			_patches.push_back(p1); _patches.push_back(p2);
			_patches.push_back(p3); _patches.push_back(p4);
			_patches.push_back(p5); _patches.push_back(p6);
			_patches.push_back(p7); _patches.push_back(p8);
			_patches.push_back(p9);
		}
		row_p = row;
		row = row_n;
	}
	_distpatches.shrink_to_fit();
	auto distpatches = Mat(i, 9, CV_8U, _distpatches.data());
	auto patches     = Mat(X*Y, 9, CV_8U, _patches.data());
	printf("%.1f%% of patches considered distinct\n", 100.f * (float)i  / (float)(X*Y));


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


	/*******************/
	/* Post-processing */
	/*******************/

	// Dilate-then-erode to close holes
	Mat out_closed;
	morphologyEx(out, out_closed, MORPH_CLOSE,
	             getStructuringElement(MORPH_RECT, Size(3, 3)));

	// Normalise then return
	Mat out_norm;
	normalize(out_closed, out_norm, 0.f, 1.f, NORM_MINMAX);
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
	for (uint idx = 0, j = 0; j < H; j++)
		for (uint i = 0; i < W; i++)
		{
			idx = segmentation[j*W + i];
			spxl_cols[idx][0] += (float)img.ptr<Vec3b>(j)[i][0];
			spxl_cols[idx][1] += (float)img.ptr<Vec3b>(j)[i][1];
			spxl_cols[idx][2] += (float)img.ptr<Vec3b>(j)[i][2];
			spxl_cnts[idx]++;
		}

	// Divide by no. of pixels
	for (uint i = 0; i < spxl_n; i++)
	{
		spxl_cols[i][0] /= spxl_cnts[i];
		spxl_cols[i][1] /= spxl_cnts[i];
		spxl_cols[i][2] /= spxl_cnts[i];
	}

	// 2. Aggregate colour distances
	auto spxl_dist = std::vector<float>(spxl_n);
	float dist, weight;
	for (uint i1 = 0; i1 < spxl_n; i1++)
	{
		if (spxl_cnts[i1] == 0) continue;
		dist = 0.f;
		for (uint i2 = i1 + 1; i2 < spxl_n; i2++) {
			if (spxl_cnts[i2] == 0) continue;

			weight = norm(spxl_cols[i1] - spxl_cols[i2]);

			dist += weight;
			spxl_dist[i2] += weight;
		}
		spxl_dist[i1] += dist;
	}

	// 3. Assign distance value to output colour distinctiveness map
	auto out = Mat(img.size(), CV_32F);
	for (uint idx = 0, j = 0; j < H; j++)
		for (uint i = 0; i < W; i++)
		{
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
	Mat out = Mat::zeros(D.size(), CV_32F);

	float thresh = 0.f, v, M;
	Vec2f CoM;
	for (uint i = 0; i < 10; i++)
	{
		M      = 0.f;
		CoM[0] = 0.f;
		CoM[1] = 0.f;
		for (uint y = 0; y < out.rows; y++)
			for (uint x = 0; x < out.cols; x++)
			{
				v = D.ptr<float>(y)[x];
				if (v > thresh)
				{
					CoM[0] += v * (float)x;
					CoM[1] += v * (float)y;
					M += v;
				}
			}

		addGaussian(out, round(CoM[0] / M), round(CoM[1] / M), 10000, thresh);
		thresh += 0.1f;
	}

	// Add centre prior
	addGaussian(out, out.cols / 2, out.rows / 2, 10000, 5);

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
const float maxSize = 800.f;

Mat getSaliency(const Mat& img)
{
	Mat img_BGR_1;

	uint H  = img.rows,
	     W  = img.cols,
	     HW = H * W;

	// Scale image to have not more than maxSize pixels on its larger
	// dimension
	float scale = (float) max(H, W) / maxSize;
	resize(img, img_BGR_1, Size(W / scale, H / scale));
	W = W / scale;
	H = H / scale;

	Mat img_BGR_2, img_BGR_4;
	resize(img_BGR_1, img_BGR_2, Size(W / 2, H / 2));
	resize(img_BGR_1, img_BGR_4, Size(W / 4, H / 4));

	// Get grayscale image to work on
	auto img_grey_1 = Mat(img_BGR_1.size(), img_BGR_1.type());
	auto img_grey_2 = Mat(img_BGR_2.size(), img_BGR_2.type());
	auto img_grey_4 = Mat(img_BGR_4.size(), img_BGR_4.type());
	cvtColor(img_BGR_1, img_grey_1, CV_BGR2GRAY);
	cvtColor(img_BGR_2, img_grey_2, CV_BGR2GRAY);
	cvtColor(img_BGR_4, img_grey_4, CV_BGR2GRAY);

	auto img_lab_1 = Mat(img_BGR_1.size(), img_BGR_1.type());
	auto img_lab_2 = Mat(img_BGR_2.size(), img_BGR_2.type());
	auto img_lab_4 = Mat(img_BGR_4.size(), img_BGR_4.type());
	cvtColor(img_BGR_1, img_lab_1, CV_BGR2Lab);
	cvtColor(img_BGR_2, img_lab_2, CV_BGR2Lab);
	cvtColor(img_BGR_4, img_lab_4, CV_BGR2Lab);

	// Get SLIC superpixels
	auto segmentation_1 = std::vector<vl_uint32>(H*W);
	auto segmentation_2 = std::vector<vl_uint32>(H*W/4);
	auto segmentation_4 = std::vector<vl_uint32>(H*W/16);
	_getSLICSegments(img_lab_2, segmentation_2);
	_getSLICSegments(img_lab_4, segmentation_4);
	_getSLICSegments(img_lab_1, segmentation_1); // Out-of-order for vis purposes

	// Calculate variance of super pixels
	auto spxl_n_1 = std::accumulate(segmentation_1.begin(),
		segmentation_1.end(), 0, [&](vl_uint32 b, vl_uint32 n) {
			return n > b ? n : b;
		}) + 1;
	auto spxl_n_2 = std::accumulate(segmentation_2.begin(),
		segmentation_2.end(), 0, [&](vl_uint32 b, vl_uint32 n) {
			return n > b ? n : b;
		}) + 1;
	auto spxl_n_4 = std::accumulate(segmentation_4.begin(),
		segmentation_4.end(), 0, [&](vl_uint32 b, vl_uint32 n) {
			return n > b ? n : b;
		}) + 1;
	printf("\nCalculated %d, %d, %d superpixels.\n", spxl_n_1, spxl_n_2, spxl_n_4);
	auto spxl_vars_1  = std::vector<float>(spxl_n_1);
	auto spxl_vars_2  = std::vector<float>(spxl_n_2);
	auto spxl_vars_4  = std::vector<float>(spxl_n_4);
	auto var_thresh_1 = _getSLICVariances(img_grey_1, segmentation_1, spxl_vars_1);
	auto var_thresh_2 = _getSLICVariances(img_grey_2, segmentation_2, spxl_vars_2);
	auto var_thresh_4 = _getSLICVariances(img_grey_4, segmentation_4, spxl_vars_4);

	// Compute pattern distinctiveness maps
	Mat patternD_1 = _getPatternDistinct(img_grey_1, segmentation_1, spxl_vars_1, var_thresh_1);
	Mat patternD_2 = _getPatternDistinct(img_grey_2, segmentation_2, spxl_vars_2, var_thresh_2);
	Mat patternD_4 = _getPatternDistinct(img_grey_4, segmentation_4, spxl_vars_4, var_thresh_4);
	Mat patternD_2_, patternD_4_;
	resize(patternD_2, patternD_2_, patternD_1.size());
	resize(patternD_4, patternD_4_, patternD_1.size());
	Mat patternD = (patternD_1 + patternD_2_ + patternD_4_) / 3;
	showImage("Pattern Distinctiveness", patternD);

	// Compute colour distinctiveness maps
	Mat colourD_1 = _getColourDistinct(img_lab_1, segmentation_1, spxl_n_1);
	Mat colourD_2 = _getColourDistinct(img_lab_2, segmentation_2, spxl_n_2);
	Mat colourD_4 = _getColourDistinct(img_lab_4, segmentation_4, spxl_n_4);
	Mat colourD_2_, colourD_4_;
	resize(colourD_2, colourD_2_, colourD_1.size());
	resize(colourD_4, colourD_4_, colourD_1.size());
	Mat colourD = (colourD_1 + colourD_2_ + colourD_4_) / 3;
	showImage("Colour Distinctiveness", colourD);

	// Calculate distinctiveness map from pattern and colour distinctiveness
	Mat D = colourD.mul(patternD);
	showImage("Distinctiveness", D);

	// Compute Gaussian weight map to highlight clusters
	Mat G = _getWeightMap(D);
	showImage("Gaussian Weight Map", G);

	// Final Saliency Map
	Mat out;
	normalize(D.mul(G), out, 0.f, 1.f, NORM_MINMAX);
	//showImage("Saliency Map", out);

	// Show result of Boolean Map approach
	// 2013. Zhang and Sclaroff
	auto bms = BMS(img_lab_1, 3, true, false, CL_Lab, false);
	bms.computeSaliency(4);
	showImage("Boolean Map", bms.getSaliencyMap());

	// Scale back to original size for further processing
	Mat out_scaled = Mat();
	resize(out, out_scaled, img.size());
	std::swap(out, out_scaled);

	return out;
}
