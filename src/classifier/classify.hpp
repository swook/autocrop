class Classifier
{
public:
	Classifier();

	// TODO: Load SVM model from file

	// TODO: Classify using given image
private:
	cv::Ptr<cv::ml::SVM> model;
};
