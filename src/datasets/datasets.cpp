#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

#include "cvmatio/MatlabIO.hpp"

#include "datasets.hpp"
#include "../util.hpp"
#include "../saliency/saliency.hpp"
#include "../features/feature.hpp"

// DataSets namespace
namespace ds
{
	DataSet::DataSet()
	{
		data = std::vector<Entry>();
	}

	void DataSet::addToTrainer(Trainer& trainer) {}

	Chen::Chen() : DataSet() {}

	/**
	 * Crop coordinates given by datasets can be noisy with some examples being
	 * - Zero width/height crops
	 * - Negative starting coordinates
	 */
	Rect Chen::getFixedCrop(const Mat& img, const Mat& crop)
	{
		int y    = crop.at<double>(0),
		    x    = crop.at<double>(1),
		    ymax = crop.at<double>(2),
		    xmax = crop.at<double>(3);

		// Correct bad data
		y = y < 0 ? 0 : y;
		x = x < 0 ? 0 : x;
		ymax = ymax < 0 ? 0 : ymax;
		xmax = xmax < 0 ? 0 : xmax;
		ymax = ymax >= img.rows ? img.rows - 1 : ymax;
		xmax = xmax >= img.cols ? img.cols - 1 : xmax;
		int h    = ymax - y,
		    w    = xmax - x;

		std::cout << "Current crop: " << crop << std::endl;
		if (h < 1 && w < 1)
		{
			std::cout << "Bad crop: " << crop << std::endl;
			throw std::runtime_error("Invalid crop");
		}

		return Rect(x, y, w, h);
	}


	void Chen::addToTrainer(Trainer& trainer)
	{
		MatlabIO mio;
		mio.open("../datasets/Chen/500_image_dataset.mat", "r");

		auto MAT = mio.read();
		mio.close();

		auto img_gt = MAT[0].data<std::vector<std::vector<MatlabIOContainer>>>();

#pragma omp parallel for
		//for (int i = 0; i < img_gt.size(); i++)
		for (int i = 0; i < 10; i++)
		{
			auto path = img_gt[i][0].data<std::string>();

			std::cout << "Loading: " << path  << " (" << i << "/" <<
				img_gt.size() << ")" << std::endl;

			Mat mat = img_gt[i][1].data<Mat>();
			Mat img = imread("../datasets/Chen/image/" + path);

			Mat grey;
			cvtColor(img, grey, CV_BGR2GRAY);

			Mat saliency = getSaliency(img);
			Mat grad     = getGrad(grey);

			for (int c = 0; c < mat.rows; c++)
			{
				// Mechanical turk data is good crop
				Rect crop;
				try {
					crop = getFixedCrop(grey, mat.row(c));
				} catch (std::exception e) {
					std::cout << "Invalid crop: " << mat.row(c) << std::endl;
					continue;
				}
				trainer.add(saliency, grad, crop, GOOD_CROP);

				// Randomly generated crop is "bad"
				// TODO: Make sure overlap with good crop is not large
				trainer.add(saliency, grad, randomCrop(saliency), BAD_CROP);
			}
		}

	}
}
