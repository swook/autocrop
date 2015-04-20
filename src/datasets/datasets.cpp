#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

#include "cvmatio/MatlabIO.hpp"

#include "datasets.hpp"
#include "../util.hpp"
#include "../saliency/saliency.hpp"
#include "../autocrop/feature.hpp"

// DataSets namespace
namespace ds
{
	DataSet::DataSet()
	{
		data = std::vector<Entry>();
	}

	void DataSet::addToTrainer(Trainer& trainer) {}

	Chen::Chen() : DataSet() {}

	void Chen::addToTrainer(Trainer& trainer)
	{
		MatlabIO mio;
		mio.open("../datasets/Chen/500_image_dataset.mat", "r");

		auto MAT = mio.read();
		mio.close();

		auto img_gt = MAT[0].data<std::vector<std::vector<MatlabIOContainer>>>();

		std::string path;
		Mat mat, img, grey, saliency, grad, feats;

		for (int i = 0; i < img_gt.size(); i++)
		{
			path = img_gt[i][0].data<std::string>();
			mat  = img_gt[i][1].data<Mat>();

			std::cout << "Loading: " << path << std::endl;
			img = imread("../datasets/Chen/image/" + path);
			cvtColor(img, grey, CV_BGR2GRAY);

			saliency = getSaliency(img);
			grad     = getGrad(grey);

			for (int c = 0; c < mat.rows; c++)
				trainer.add(saliency, grad, mat.row(c), GOOD_CROP);
		}

	}
}
