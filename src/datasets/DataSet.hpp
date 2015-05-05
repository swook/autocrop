#pragma once

#include "opencv2/core.hpp"
#include "../features/FeatMat.hpp"

// A namespace is used to contain all dataset related classes/methods
namespace ds
{

	/**
	 * A DataSet contains details to acquire images (including candidate
	 * crops) with associated classifications.
	 *
	 * The classification is 0|1 where 1 is a good crop.
	 */
	class DataSet
	{
	public:
		/**
		 * Traverses dataset and adds all available features to a given
		 * FeatMat
		 */
		virtual void addToFeatMat(FeatMat& featMat) {};
	};
}
