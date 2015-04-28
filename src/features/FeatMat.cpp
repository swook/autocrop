#include <iostream>

#include "opencv2/core.hpp"
using namespace cv;

#include "FeatMat.hpp"
#include "feature.hpp"
#include "../util/file.hpp"

FeatMat::FeatMat()
{
	feats = Mat(Size(FEATS_N, 0), CV_32F);
	resps = Mat(Size(      1, 0), CV_32F);
}

void FeatMat::addFeatVec(const Mat featVec, const int cls)
{
	assert(featVec.cols == feats.cols);

	// Add vector (with class) to matrix
	mtx.lock();

	feats.push_back(featVec);
	resps.push_back(cls);

	mtx.unlock();
}

void FeatMat::save(std::string fpath)
{
	FileStorage fs = FileStorage(fpath, FileStorage::WRITE | FileStorage::FORMAT_YAML);

	mtx.lock();
	write(fs, "feats", feats);
	write(fs, "resps", resps);
	mtx.unlock();
}

void FeatMat::load(std::string fpath)
{
	FileStorage fs = FileStorage(fpath, FileStorage::READ | FileStorage::FORMAT_YAML);

	mtx.lock();
	fs["feats"] >> feats;
	fs["resps"] >> resps;
	mtx.unlock();

	assert(feats.cols == FEATS_N);
	assert(resps.cols == 1);
}

