#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"
using namespace cv;

#include "EarthPorn.hpp"
#include "../constants.hpp"
#include "../util/math.hpp"
#include "../util/file.hpp"
#include "../features/feature.hpp"
#include "../features/FeatMat.hpp"


// DataSets namespace
namespace ds
{

	void EarthPorn::addToFeatMat(FeatMat& featMat)
	{
		paths files = getUnprocessedImagePaths("../datasets/EarthPorn/");
#pragma omp parallel for
		for (int i = 0; i < files.size(); i++)
		{
			auto fpath = files[i].string();

			std::cout << "Loading: " << fpath << " (" << i << "/" <<
				files.size() << ")" << std::endl;

			// Load cached image maps. Abort if invalid image [maps]
			Mat saliency, grad;
			try {
				saliency = imread(setSuffix(fpath, "saliency").string(), CV_LOAD_IMAGE_UNCHANGED);
				grad     = imread(setSuffix(fpath, "grad").string(), CV_LOAD_IMAGE_UNCHANGED);
			}
			catch (std::exception e) {
				std::cout << "Error reading: " << fpath << std::endl;
				continue;
			}
			if (!saliency.data || !grad.data) continue;

			// Image is good crop
			Rect crop = Rect(0, 0, saliency.cols, saliency.rows);
			featMat.addFeatVec(saliency, grad, crop, GOOD_CROP);


			// Randomly generated crop is "bad"
			// TODO: Add multiple
			featMat.addFeatVec(saliency, grad, randomCrop(saliency), BAD_CROP);
			featMat.addFeatVec(saliency, grad, randomCrop(saliency), BAD_CROP);
			featMat.addFeatVec(saliency, grad, randomCrop(saliency), BAD_CROP);
			featMat.addFeatVec(saliency, grad, randomCrop(saliency), BAD_CROP);
		}

	}
}

