#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"
using namespace cv;

#include "cvmatio/MatlabIO.hpp"

#include "Chen.hpp"
#include "../constants.hpp"
#include "../autocrop/autocrop.hpp"
#include "../classify/Classifier.hpp"
#include "../util/math.hpp"
#include "../util/file.hpp"
#include "../util/opencv.hpp"
#include "../features/feature.hpp"
#include "../features/FeatMat.hpp"


// DataSets namespace
namespace ds
{

	void Chen::addToFeatMat(FeatMat& featMat)
	{
		getTurkCrops();

#pragma omp parallel for
		for (int i = 0; i < turkCrops.size(); i++)
		{
			auto fname = turkCrops[i].fname;
			auto crops = turkCrops[i].crops;

			printf("Loading %s (%d/%d)\n", fname.c_str(), i + 1,
				turkCrops.size());

			Mat saliency, gradient;
			try {
				auto maps = getMaps(fname);
				saliency = maps.first;
				gradient = maps.second;
			} catch (std::exception e) {
				printf("Error in loading %s", fname.c_str());
				continue;
			}

			Rect bad_crop, good_crop;

			// Image is bad crop
			bad_crop = Rect(0, 0, saliency.cols, saliency.rows);
			featMat.addFeatVec(saliency, gradient, bad_crop, BAD_CROP);

			// For each Mechanical Turk crop for given image
			for (int c = 0; c < crops.rows; c++)
			{
				// Mechanical turk data is good crop
				try {
					good_crop = getFixedCrop(saliency, crops.row(c));
				} catch (std::exception e) {
					std::cout << "Invalid crop: " << crops.row(c) << std::endl;
					continue;
				}
				featMat.addFeatVec(saliency, gradient, good_crop, GOOD_CROP);

				// Randomly generated crop is "bad"
				// Only picks crop with less than 0.4 overlap
				bad_crop = randomCrop(saliency, good_crop, 0.3);
				featMat.addFeatVec(saliency, gradient, bad_crop, BAD_CROP);
			}
		}
	}

	void Chen::quantEval()
	{
		getTurkCrops();

		// Initialise classifier
		Classifier classifier;
		classifier.loadModel("Trained_model.yml");

		const int MAX_CROP_CANDS = 10;
		std::vector<float> indices;
		std::vector<std::vector<float>> overlaps(10);

		const int N = turkCrops.size();

#pragma omp parallel for
		for (int i = 0; i < N; i++)
		{
			auto fname = turkCrops[i].fname;
			auto _crops = turkCrops[i].crops;

			Mat saliency, gradient;
			try {
				auto maps = getMaps(fname);
				saliency = maps.first;
				gradient = maps.second;
			} catch (std::exception e) {
				printf("Error in loading %s\n", fname.c_str());
				continue;
			}

			Candidates crop_cands = getCropCandidates(classifier, saliency, gradient);

			// Get list of valid crops
			std::vector<Rect> crops;

			Rect crop;
			int best_crop;
			float overlap, max_overlap = 0.f;

			for (int C = 1; C <= MAX_CROP_CANDS; C++)
			{
				for (int a = 0; a < C; a++)
					for (int c = 0; c < _crops.rows; c++)
					{
						// Mechanical turk data is good crop
						try {
							crop = getFixedCrop(saliency, _crops.row(c));
							crops.push_back(crop);
						} catch (std::exception e) {
							continue;
						}

						overlap = cropOverlap(crop_cands[a]->crop, crop);

						max_overlap = max(max_overlap, overlap);
						if (max_overlap == overlap)
							best_crop = a;
					}

				overlaps[C-1].push_back(max_overlap);
			}

			indices.push_back(best_crop);
			printf("[%3d/%3d] Best crop index: %2d, Max overlap: %.2f, Filename: %s\n",
				i, N, best_crop, max_overlap, fname.c_str());
		}

		printf("\n%d images were evaluated.\n", N);

		for (int C = 1; C <= 5; C++)
		{
			printf("\nFor %d top candidates only:\n", C);
			printf("- Mean max overlap is: %.3f\n", mean(overlaps[C-1]));
			printf("- Median max overlap is: %.3f\n\n", median(overlaps[C-1]));
		}
		printf("- Mean best crop index is: %.1f\n", mean(indices));
		printf("- Median best crop index is: %.1f\n", median(indices));
	}

	void Chen::getTurkCrops()
	{
		// Reset list
		std::vector<TurkCrop>().swap(turkCrops);

		// Read mat file with Mechanical Turk sourced crops
		MatlabIO mio;
		mio.open("../datasets/Chen/500_image_dataset.mat", "r");
		auto MAT = mio.read();
		mio.close();

		auto img_gt = MAT[0].data<std::vector<std::vector<MatlabIOContainer>>>();

		for (int i = 0; i < img_gt.size(); i++)
			turkCrops.push_back((TurkCrop) {
				img_gt[i][0].data<std::string>(),
				img_gt[i][1].data<Mat>()
			});
	}


	std::pair<Mat, Mat> Chen::getMaps(std::string fname)
	{
		// Load image. Abort if invalid image [maps]
		Mat img = imread("../datasets/Chen/image/" + fname);
		if (!img.data)
			throw std::runtime_error("Chen: Error in loading image maps");

		Mat saliency = getSaliency(img),
		    gradient = getGradient(img);

		return std::pair<Mat, Mat>(saliency, gradient);
	}


	Rect Chen::getFixedCrop(const Mat& img, const Mat& crop)
	{
		int y    = crop.at<double>(0),
		    x    = crop.at<double>(1),
		    ymax = crop.at<double>(2),
		    xmax = crop.at<double>(3);

		// Correct bad data
		y = y < 0 ? 0 : y;
		x = x < 0 ? 0 : x;
		ymax = ymax < 0 ? 0 : ymax;
		xmax = xmax < 0 ? 0 : xmax;
		ymax = ymax >= img.rows ? img.rows - 1 : ymax;
		xmax = xmax >= img.cols ? img.cols - 1 : xmax;
		int h = ymax - y,
		    w = xmax - x;

		//std::cout << "Current crop: " << crop << std::endl;
		if (h < 2 || w < 2)
		{
			//std::cout << "Bad crop: " << crop << std::endl;
			throw std::runtime_error("Invalid crop");
		}

		return Rect(x, y, w, h);
	}
}

