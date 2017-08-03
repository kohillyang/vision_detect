#ifndef DIGITAL_CLASSIFICATION_HPP_
#define DIGITAL_CLASSIFICATION_HPP_


#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


typedef std::pair<string, float> Prediction;

class Classifier
{
public:
	Classifier(const string& model_file, const string& trained_file,
			const string& mean_file, const string& label_file);

	Classifier();

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

	string classify_mnist(const cv::Mat& IMG);




 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif
