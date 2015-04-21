#pragma once

#include "opencv2/core.hpp"
#include "../training/train.hpp"

// A namespace is used to contain all dataset related classes/methods
namespace ds
{
	// Classes available
	enum
	{
		GOOD_CROP = 1,
		BAD_CROP  = 0,
	};

	// Representation of stored data
	struct Entry
	{
		const cv::Mat saliency;
		const cv::Mat grad;
		const cv::Mat crop;
		const int     cls;
	};
	typedef std::vector<Entry> Entries;


	// A DataSet contains details to acquire images (including candidate
	// crops) with associated classifications.
	//
	// The classification is 0|1 where 1 is a good crop.
	class DataSet
	{
	public:
		DataSet();
		void addToTrainer(Trainer& trainer);

		Entries data;
	};


	// Dataset from (2014) Chen et al. Automatic Image Cropping using Visual
	// Composition, Boundary Simplicity and Content Preservation Models.
	//
	// Composed of images and crop windows collected using Amazon Mechanical
	// Turk.
	class Chen : public DataSet
	{
	public:
		Chen();
		cv::Rect getFixedCrop(const cv::Mat& img, const cv::Mat& crop);
		void     addToTrainer(Trainer& trainer);
	};
}
