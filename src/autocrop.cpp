#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "autocrop/autocrop.hpp"
#include "saliency/saliency.hpp"
#include "features/feature.hpp"
#include "util/opencv.hpp"

int main(int argc, char** argv)
{
	/*
	 * Argument parsing
	 */
	po::options_description desc("Available options");
	desc.add_options()
	    ("help", "Show this message")
	    ("aspect-ratio,r", po::value<float>()->default_value(0.f), "Width-to-height ratio")
	    ("input-file,i", po::value<std::string>(), "Input file path")
	    ("output-file,o", po::value<std::string>(), "Output file path (default: output.png)")
	    ("headless,hl", po::bool_switch()->default_value(false), "Run without graphical output")
	;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.size() == 0 || vm.count("help") || !vm.count("input-file")) {
		std::cout << "Usage: " << argv[0]
			<< " [options] input-file" << std::endl
			<< desc;
		return 1;
	}

	if (vm["headless"].as<bool>()) GRAPHICAL = false;

	/*
	 * Read image file
	 */
	std::string const f = vm["input-file"].as<std::string>();
	Mat const in = imread(f, CV_LOAD_IMAGE_COLOR);
	if (!in.data) {
		throw std::runtime_error("Invalid input file: " + f);
		return -1;
	}

	/*
	 * Call retargeting methods
	 */
	Mat saliency = getSaliency(in);
	Mat gradient = getGradient(in);
	Rect crop = getBestCrop(saliency, gradient, vm["aspect-ratio"].as<float>());

	// Set crop border pixels to red
	Mat in_crop = in.clone();
	Scalar red = Scalar(0, 0, 255);
	in_crop(Rect(crop.x, crop.y, crop.width, 1)) = red; // Top
	in_crop(Rect(crop.x+crop.width-1, crop.y, 1, crop.height)) = red; // Right
	in_crop(Rect(crop.x, crop.y+crop.height-1, crop.width, 1)) = red; // Bottom
	in_crop(Rect(crop.x, crop.y, 1, crop.height)) = red; // Left

	// Set cropped out region black
	Mat out_crop = in.clone();
	Scalar black = Scalar(0, 0, 0);
	out_crop(Rect(0, 0, in.cols, crop.y)) = black; // Top
	out_crop(Rect(crop.x+crop.width, 0, in.cols-crop.x-crop.width, in.rows)) = black; // Right
	out_crop(Rect(0, crop.y+crop.height, in.cols, in.rows-crop.y-crop.height)) = black; // Bottom
	out_crop(Rect(0, 0, crop.x, in.rows)) = black; // Left

	// Show saliency and crop side-by-side
	const Mat out = my_hconcat({saliency, in_crop, out_crop});

	// Show output image
	showImageAndWait("Input - Cropped", out);

	/*
	 * Save output if necessary
	 */
	if (vm.count("output-file")) {
		imwrite(vm["output-file"].as<std::string>(), out);
	}

	return 0;

}
