#pragma once

#include <mutex>
#include "opencv2/core.hpp"

class FeatMat
{
public:
	FeatMat();
	void addFeatVec(const cv::Mat featVec, const int cls);

	void save(std::string fpath);
	void load(std::string fpath);

private:
	cv::Mat feats;
	cv::Mat resps;

	std::mutex mtx;
};
