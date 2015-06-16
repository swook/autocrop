#pragma once

#include "opencv2/core.hpp"
#include "DataSet.hpp"
#include "../features/FeatMat.hpp"

namespace ds
{
	/**
	 * Dataset from https://www.reddit.com/r/Reddit
	 * Top images from the past year are downloaded.
	 * See datasets/Reddit/get_dataset.py
	 *
	 * The whole image is considered to be well composed,
	 * and random crops are considered to be badly composed.
	 */
	class Reddit : public DataSet
	{
	public:
		void addToFeatMat(FeatMat& featMat);
	};
}
