#pragma once

#include <map>

#include "opencv2/core.hpp"
#include "DataSet.hpp"
#include "../features/FeatMat.hpp"

// A namespace is used to contain all dataset related classes/methods
namespace ds
{

	/**
	 * Dataset from (2014) Chen et al. Automatic Image Cropping using Visual
	 * Composition, Boundary Simplicity and Content Preservation Models.
	 *
	 * Composed of images and crop windows collected using Amazon Mechanical
	 * Turk.
	 */
	class Chen : public DataSet
	{
	public:
		void addToFeatMat(FeatMat& featMat);
	private:

		struct TurkCrop
		{
			std::string fname;
			cv::Mat     crops;
		};
		std::vector<TurkCrop> turkCrops;


		/**
		 * Get Mechanical Turk crops from 500_image_dataset.mat and
		 * store into turkCrops
		 */
		void getTurkCrops();


		/**
		 * Get saliency and gradient maps for a given file name
		 */
		std::pair<cv::Mat, cv::Mat> getMaps(std::string fname);


		/**
		 * Crop coordinates given by datasets can be noisy with some examples being
		 * - Zero width/height crops
		 * - Negative starting coordinates
		 */
		cv::Rect getFixedCrop(const cv::Mat& img, const cv::Mat& crop);
	};
}
