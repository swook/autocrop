#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "opencv2/core.hpp"
using namespace cv;

#include "datasets/datasets.hpp"
#include "features/feature.hpp"
#include "features/FeatMat.hpp"
#include "util/opencv.hpp"
#include "util/file.hpp"

int main(int argc, char** argv)
{
	GRAPHICAL = false;

	/**
	 * Process datasets into single training matrix
	 * TODO: Add more datasets
	 * TODO: Reserve memory for feature matrix
	 */
	FeatMat featMat;

	// (2014) Chen et al. Automatic Image Cropping using Visual Composition,
	// Boundary Simplicity and Content Preservation Models.
	ds::Chen chen;
	chen.addToFeatMat(featMat);

	// Reddit
	ds::Reddit reddit;
	reddit.addToFeatMat(featMat);

	// Save feature matrix
	featMat.save("./Training.yml");

	return 0;
}
