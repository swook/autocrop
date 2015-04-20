#pragma once

#include "opencv2/core.hpp"

// A namespace is used to contain all dataset related classes/methods
namespace ds
{
	// Status codes for parsing in dataset
	enum
	{
		EOS = 1, // End of set
	};

	// A DataSet contains details to acquire images (including candidate
	// crops) with associated classifications.
	//
	// The classification is 0|1 where 1 is a good crop.
	class DataSet
	{
		DataSet();
		void init();
		cv::Mat getRow();
		cv::Mat getMat();
	};


	// Dataset from (2014) Chen et al. Automatic Image Cropping using Visual
	// Composition, Boundary Simplicity and Content Preservation Models.
	//
	// Composed of images and crop windows collected using Amazon Mechanical
	// Turk.
	class Chen : DataSet
	{
	};
}
