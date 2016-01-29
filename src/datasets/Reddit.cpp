#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"
using namespace cv;

#include "Reddit.hpp"
#include "../constants.hpp"
#include "../util/math.hpp"
#include "../util/file.hpp"
#include "../util/opencv.hpp"
#include "../saliency/saliency.hpp"
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
			auto fpath = files[i];

			std::cout << "Loading: " << fpath.string() << " (" << i
				<< "/" << files.size() << ")" << std::endl;

			// Load image. Abort if invalid image [maps]
			Mat saliency, gradient;
			try {
				saliency = imread(setSuffix(fpath, "saliency").replace_extension(".exr").string(), CV_LOAD_IMAGE_UNCHANGED);
				gradient = imread(setSuffix(fpath, "gradient").replace_extension(".exr").string(), CV_LOAD_IMAGE_UNCHANGED);
				if (!saliency.data || !gradient.data)
					throw std::runtime_error("Failed loading maps");
			} catch (std::exception e) {
				Mat img = imread_reduced(fpath.string());
				if (!img.data)
				{
					std::cout << "Error reading: " << fpath << std::endl;
					continue;
				}
				saliency = getSaliency(img),
				gradient = getGradient(img);
			}

			// Image is good crop
			Rect crop = Rect(0, 0, saliency.cols, saliency.rows);
			featMat.addFeatVec(saliency, gradient, crop, GOOD_CROP);

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
					featMat.addFeatVec(saliency, gradient, crop, BAD_CROP);
					n_bad++;
				}
				if (n_bad == 2) break;
			}
		}

	}
}

