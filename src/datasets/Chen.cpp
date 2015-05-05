#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"
using namespace cv;

#include "cvmatio/MatlabIO.hpp"

#include "Chen.hpp"
#include "../constants.hpp"
#include "../util/math.hpp"
#include "../util/file.hpp"
#include "../features/feature.hpp"
#include "../features/FeatMat.hpp"


Rect getFixedCrop(const Mat& img, const Mat& crop);


// DataSets namespace
namespace ds
{

	void Chen::addToFeatMat(FeatMat& featMat)
	{
		MatlabIO mio;
		mio.open("../datasets/Chen/500_image_dataset.mat", "r");

		auto MAT = mio.read();
		mio.close();

		auto img_gt = MAT[0].data<std::vector<std::vector<MatlabIOContainer>>>();

#pragma omp parallel for
		for (int i = 0; i < img_gt.size(); i++)
		{
			auto fn = img_gt[i][0].data<std::string>();

			std::cout << "Loading: " << fn << " (" << i << "/" <<
				img_gt.size() << ")" << std::endl;

			Mat mat = img_gt[i][1].data<Mat>();
			path ipath = path("../datasets/Chen/image/" + fn);

			// Load cached image maps. Abort if invalid image [maps]
			Mat saliency, grad;
			try {
				saliency = imread(setSuffix(ipath, "saliency").string(), CV_LOAD_IMAGE_UNCHANGED);
				grad     = imread(setSuffix(ipath, "grad").string(), CV_LOAD_IMAGE_UNCHANGED);
			}
			catch (std::exception e) {
				std::cout << "Error reading: " << ipath << std::endl;
				continue;
			}
			if (!saliency.data || !grad.data) continue;

			// Image is bad crop
			Rect crop = Rect(0, 0, saliency.cols, saliency.rows);
			featMat.addFeatVec(saliency, grad, crop, BAD_CROP);

			// For each Mechanical Turk crop for given image
			for (int c = 0; c < mat.rows; c++)
			{
				// Mechanical turk data is good crop
				try {
					crop = getFixedCrop(saliency, mat.row(c));
				} catch (std::exception e) {
					std::cout << "Invalid crop: " << mat.row(c) << std::endl;
					continue;
				}
				featMat.addFeatVec(saliency, grad, crop, GOOD_CROP);

				// Randomly generated crop is "bad"
				// TODO: Make sure overlap with good crop is not large
				featMat.addFeatVec(saliency, grad, randomCrop(saliency), BAD_CROP);
			}
		}

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

