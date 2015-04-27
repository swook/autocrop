#include "opencv2/opencv.hpp"
using namespace cv;

#include "opencv.hpp"

bool GRAPHICAL = true;

Mat my_hconcat(const Mats& imgs)
{
	int h = imgs[0].rows;

	Mats outs;
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat img = imgs[i];
		Mat out = img;

		// If gray-scale, make BGR
		if (img.channels() == 1)
		{
			cvtColor(img, out, CV_GRAY2BGR);
		}

		// Make no. of rows match
		if (out.rows != h)
		{
			double ratio = (double) h / (double) img.rows;
			resize(out, out, Size(img.cols * ratio, h));
		}

		// Convert to CV_8UC3
		if (out.type() != CV_8UC3)
		{
			normalize(out, out, 0, 255, NORM_MINMAX);
			out.convertTo(out, CV_8UC3);
		}

		outs.push_back(out);
	}

	Mat out;
	hconcat(outs, out);
	return out;
}

void showImage(const char* title, const Mat& img)
{
	if (!GRAPHICAL) return;

	std::cout << "\nShowing image: \"" << title << "\"." << std::endl;
	namedWindow(title, CV_WINDOW_NORMAL);
	imshow(title, img);
}

void showImage(const char* title, const Mats& imgs)
{
	showImage(title, my_hconcat(imgs));
}

void showImageAndWait(const char* title, const Mat& img)
{
	if (!GRAPHICAL) return;

	showImage(title, img);
	std::cout << "Press any key to continue..." << std::endl;
	waitKey(0);
}

void showImageAndWait(const char* title, const Mats& imgs)
{
	showImageAndWait(title, my_hconcat(imgs));
}


/**
 * Adds a Gaussian of specified standard deviation centred at a specified
 * coordinate on a given image
 */
void addGaussian(Mat& img, uint x, uint y, float std, float weight)
{
	uint H  = img.rows,
	     W  = img.cols,
	     HW = H * W;

	const float a =  1.f / (std * std * 2.f * M_PI),
	            b = -.5f / (std * std);

	auto  done = std::vector<bool>(HW);  // Mark if calculation done
	float dy2, dx2, g;
	int   dy, dx, j, i, j_, i_; // Mirrored pixels to check
	uint  ji_;

	for (j = 0; j < H; j++)
	{
		dy  = abs(j - (int)y);
		dy2 = dy * dy;
		for (i = 0; i < W; i++)
		{
			if (done[j * W + i]) continue;

			dx  = abs(i - (int)x);
			dx2 = dx * dx;
			g   = weight * a * exp(b * (dx2 + dy2));

			j_ = y + dy; i_ = x + dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y - dy; i_ = x + dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y + dy; i_ = x - dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y - dy; i_ = x - dx; ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}
		}
	}
}

