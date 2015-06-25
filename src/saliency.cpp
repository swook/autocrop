#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
namespace fs = boost::filesystem;
namespace po = boost::program_options;

#include "opencv2/core.hpp"
using namespace cv;

#include "saliency/saliency.hpp"
#include "features/feature.hpp"
#include "util/opencv.hpp"
#include "util/file.hpp"

int main(int argc, char** argv)
{
	/*
	 * Argument parsing
	 */
	po::options_description desc("Available options");
	desc.add_options()
	    ("help", "Show this message")
	    ("input-path,i", po::value<std::string>(), "Input file path")
	    ("output-file,o", po::value<std::string>(), "Output file path")
	    ("output-dir,d", po::value<std::string>(), "Output directory")
	    ("headless,hl", po::bool_switch()->default_value(false), "Run without graphical output")
	;

	po::positional_options_description p;
	p.add("input-path", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.size() == 0 || vm.count("help") || !vm.count("input-path")) {
		std::cout << "Usage: " << argv[0]
			<< " [options] input-path" << std::endl
			<< desc;
		return 1;
	}

	if (vm["headless"].as<bool>()) GRAPHICAL = false;

	/*
	 * Read image file
	 *
	 * - If input-path is a file, load image and create saliency map.
	 *   Save output if output-file specified.
	 *
	 * - If input-path is a directory, load images and save output to
	 *   output-dir if specified.
	 */
	fs::path in_path = fs::path(vm["input-path"].as<std::string>());
	if (fs::is_regular_file(in_path))
	{
		// Load Image
		Mat const img = imread(in_path.native(), CV_LOAD_IMAGE_COLOR);
		if (!img.data)
		{
			throw std::runtime_error("Invalid input file: " + in_path.native());
			return -1;
		}

		// Calculate saliency map
		Mat out = getSaliency(img);

		// Show input and output image side-by-side
		showImageAndWait("Input - Output", {img, out});

		// Save output if necessary
		if (vm.count("output-file"))
		{
			Mat out_norm;
			normalize(out, out_norm, 0.f, 255.f, NORM_MINMAX);
			imwrite(vm["output-file"].as<std::string>(), out_norm);
		}
	}
	else if (fs::is_directory(in_path))
	{
		paths ins = getUnprocessedImagePaths(in_path);

		// If output-dir specified, write output files
		// Also, calculate gradient image for constructing features
		if (vm.count("output-dir"))
		{
#pragma omp parallel for
			for (int i = 0; i < ins.size(); i++)
			{
				path ipath = ins[i];
				std::string osali = vm["output-dir"].as<std::string>() + "/" +
				                    setSuffix(ins[i], "saliency").filename().string(),
				            ograd = vm["output-dir"].as<std::string>() + "/" +
				                    setSuffix(ins[i], "gradient").filename().string();

				if (fs::exists(osali)) continue;

				// Load Image
				Mat const img = imread(ins[i].string(), CV_LOAD_IMAGE_COLOR);
				if (!img.data)
				{
					std::cout << ins[i] << " is not a valid image." << std::endl;
					continue;
				}
				std::cout << "> Processing\t" << i << "/" << ins.size()
					<< ": " << ins[i] << "..." << std::endl;

				// Calculate saliency and gradient map
				Mat sali, grad, grey;
				cvtColor(img, grey, CV_BGR2GRAY);
				sali = getSaliency(img);
				grad = getGradient(grey);

				// Save calculated maps
				std::cout << "> Writing\t" << i << "/" << ins.size()
					<< ": " << osali << "..." << std::endl;
				imwrite(osali, sali);
				imwrite(ograd, grad);
			}
		}
		else
		{
			Mat sali;
			for (int i = 0; i < ins.size(); i++)
			{
				path ipath = ins[i];

				// Load Image
				Mat const img = imread(ins[i].string(), CV_LOAD_IMAGE_COLOR);
				if (!img.data)
				{
					std::cout << ins[i] << " is not a valid image." << std::endl;
					continue;
				}
				std::cout << "Processing " << ins[i] << "..." << std::endl;

				// Calculate saliency map
				sali = getSaliency(img);

				// Show saliency map
				showImageAndWait("Input - Saliency", {img, sali});
			}
		}
	}

	return 0;

}
