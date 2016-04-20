#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "classify/Classifier.hpp"
#include "util/opencv.hpp"

int main(int argc, char** argv)
{
	/*
	 * Argument parsing
	 */
	po::options_description desc("Available options");
	desc.add_options()
	    ("help", "Show this message")
	    ("input-file,i", po::value<std::string>(), "Input file path")
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


	/**
	 * Read image file
	 */
	std::string const f = vm["input-file"].as<std::string>();
	Mat const img = imread(f, CV_LOAD_IMAGE_COLOR);
	if (!img.data) {
		throw std::runtime_error("Invalid input file: " + f);
		return -1;
	}
	showImage("Input Image", img);


	/**
	 * Load trained model
	 */
	Classifier classifier;
#if FANG
	classifier.loadModel("Trained_model_Fang.yml");
#else
	classifier.loadModel("Trained_model.yml");
#endif
	float score = classifier.classifyRaw(img);
	std::cout << "Score: " << score << std::endl;
	bool good = score < 0;


	/**
	 * Print output
	 */
	if (good)
		std::cout << "The input image has good saliency composition." << std::endl;
	else
		std::cout << "The input image has bad saliency composition." << std::endl;

	return 0;

}
