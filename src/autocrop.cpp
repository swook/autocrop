#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "autocrop/autocrop.hpp"
#include "saliency/saliency.hpp"
#include "features/feature.hpp"
#include "util/file.hpp"
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
	    ("text-only,t", po::bool_switch()->default_value(false), "Only return crop as text")
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
	Mat saliency, gradient;
	try {
		saliency = imread(setSuffix(f, "saliency").string(), CV_LOAD_IMAGE_UNCHANGED);
		gradient = imread(setSuffix(f, "gradient").string(), CV_LOAD_IMAGE_UNCHANGED);
	} catch (std::exception e) {}
	if (!saliency.data) saliency = getSaliency(in);
	if (!gradient.data) gradient = getGradient(in);
	Rect crop = getBestCrop(saliency, gradient, vm["aspect-ratio"].as<float>());

	if (!vm["text-only"].as<bool>())
	{
		// Set crop border pixels to red
		Mat in_crop = in.clone();
		Scalar red = Scalar(0, 0, 255);
		in_crop(Rect(crop.x, crop.y, crop.width, 3)) = red; // Top
		in_crop(Rect(crop.x+crop.width-3, crop.y, 3, crop.height)) = red; // Right
		in_crop(Rect(crop.x, crop.y+crop.height-3, crop.width, 3)) = red; // Bottom
		in_crop(Rect(crop.x, crop.y, 3, crop.height)) = red; // Left

		// Set crop border pixels in maps to white
		Mat out_sali;
		normalize(saliency, out_sali, 0.f, 255.f, NORM_MINMAX);
		cvtColor(out_sali, out_sali, CV_GRAY2BGR);
		out_sali(Rect(crop.x, crop.y, crop.width, 3)) = red; // Top
		out_sali(Rect(crop.x+crop.width-3, crop.y, 3, crop.height)) = red; // Right
		out_sali(Rect(crop.x, crop.y+crop.height-3, crop.width, 3)) = red; // Bottom
		out_sali(Rect(crop.x, crop.y, 3, crop.height)) = red; // Left

		Mat out_grad;
		normalize(gradient, out_grad, 0.f, 255.f, NORM_MINMAX);
		cvtColor(out_grad, out_grad, CV_GRAY2BGR);
		out_grad(Rect(crop.x, crop.y, crop.width, 3)) = red; // Top
		out_grad(Rect(crop.x+crop.width-3, crop.y, 3, crop.height)) = red; // Right
		out_grad(Rect(crop.x, crop.y+crop.height-3, crop.width, 3)) = red; // Bottom
		out_grad(Rect(crop.x, crop.y, 3, crop.height)) = red; // Left

		// Set cropped out region black
		Mat out_crop = in.clone()(crop);
		double ratio = (double)in.rows / (double)crop.height;
		resize(out_crop, out_crop, Size(), ratio, ratio);

		// Show saliency and crop side-by-side
		const Mat out = my_hconcat({in_crop, out_sali, out_grad, out_crop});

		// Show output image
		showImageAndWait("Input - Cropped", out);

		/*
		 * Save output if necessary
		 */
		if (vm.count("output-file")) {
			imwrite(vm["output-file"].as<std::string>(), out);
		}
	}
	else
	{
		// Just print final crop
		std::cout << crop.x << " " << crop.y << " " << crop.width << " "
			<< crop.height << std::endl;
	}

	return 0;

}
