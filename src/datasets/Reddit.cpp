#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"
using namespace cv;

#include "Reddit.hpp"
#include "../constants.hpp"
#include "../util/math.hpp"
#include "../util/file.hpp"
#include "../features/feature.hpp"
#include "../features/FeatMat.hpp"


// DataSets namespace
namespace ds
{

	void Reddit::addToFeatMat(FeatMat& featMat)
	{
		paths files = getUnprocessedImagePaths("../datasets/Reddit/");
#pragma omp parallel for
		for (int i = 0; i < files.size(); i++)
		{
			auto fpath = files[i].string();

			std::cout << "Loading: " << fpath << " (" << i << "/" <<
				files.size() << ")" << std::endl;

			// Load cached image maps. Abort if invalid image [maps]
			Mat saliency, grad;
			try
			{
				saliency = imread(setSuffix(fpath, "saliency").string(), CV_LOAD_IMAGE_UNCHANGED);
				grad     = imread(setSuffix(fpath, "grad").string(), CV_LOAD_IMAGE_UNCHANGED);
			}
			catch (std::exception e)
			{
				std::cout << "Error reading: " << fpath << std::endl;
				continue;
			}
			if (!saliency.data)// || !grad.data)
			{
				std::cout << "Error reading: " << fpath << std::endl;
				continue;
			}

			// Image is good crop
			Rect crop = Rect(0, 0, saliency.cols, saliency.rows);
			featMat.addFeatVec(saliency, grad, crop, GOOD_CROP);

			// Randomly generated crop is "bad"
			float sum_saliency = sum(saliency)[0];
			float total_area = saliency.cols * saliency.rows;
			int n_bad = 0;
			while (true)
			{
				crop = randomCrop(saliency);
				float S_content = sum(saliency(crop))[0] / sum_saliency;
				float S_area = (float)(crop.x*crop.y) / total_area;
				if (S_content < 0.2 && S_area > 0.2)
				{
					featMat.addFeatVec(saliency, grad, crop, BAD_CROP);
					n_bad++;
				}
				if (n_bad == 2) break;
			}
		}

	}
}

