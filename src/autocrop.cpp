#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "autocrop/autocrop.hpp"
#include "saliency/saliency.hpp"
#include "features/feature.hpp"
#include "util.hpp"

int main(int argc, char** argv)
{
	/*
	 * Argument parsing
	 */
	po::options_description desc("Available options");
	desc.add_options()
	    ("help", "Show this message")
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
	Mat const img = imread(f, CV_LOAD_IMAGE_COLOR);
	if (!img.data) {
		throw std::runtime_error("Invalid input file: " + f);
		return -1;
	}
	showImage("Input Image", img);

	/*
	 * Call retargeting methods
	 */
	Mat out = crop(img);

	// Show output image
	showImageAndWait("Output Image", out);

	/*
	 * Save output if necessary
	 */
	if (vm.count("output-file")) {
		Mat out_norm;
		normalize(out, out_norm, 0.f, 255.f, NORM_MINMAX);
		imwrite(vm["output-file"].as<std::string>(), out_norm);
	}

	return 0;

}
