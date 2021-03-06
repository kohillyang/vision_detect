#ifndef CPU_ONLY
#define CPU_ONLY
#endif
#include "digital_classification.hpp"

#ifndef USE_OPENCV
#define USE_OPENCV
#endif
#ifdef USE_OPENCV

#endif  // USE_OPENCV


using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


/* Pair (label, confidence) representing a prediction. */

Classifier::Classifier()
{
	string caffe_root = "/home/kohill/mnist/";
	string model_file = caffe_root + "lenet.prototxt";
	string trained_file = caffe_root + "lenet_iter_20000.caffemodel";
	string mean_file = caffe_root + "mean.binaryproto";
	string label_file = caffe_root + "mnistlabel.txt";

#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

float getaveragegray(const cv::Mat& image)
{
	int nr = image.rows; // number of rows  hang
	int nc = image.cols * image.channels(); // total number of elements per line  lie
	float totalgray = 0;
	const uchar* data;
	for (int j = 0; j < nr; j++)
	{
		data = image.ptr < uchar > (j);
		for (int i = 0; i < nc; i++)
		{
			totalgray += (*data++);
		}
	}
	totalgray = totalgray/(nr*nc);
	totalgray = totalgray+(0-totalgray)*0.2;
	return totalgray;
}
string Classifier::classify_mnist(const cv::Mat& IMG)
{
	cv::Mat img = Mat(28, 28, 0);
	cv::Mat loIMG = IMG.clone();
	if (!IMG.empty())
	{
		cv::cvtColor(loIMG, loIMG, cv::ColorConversionCodes::COLOR_BGR2GRAY);
//		cv::Rect a = cv::Rect(loIMG.cols/3, loIMG.rows/4, loIMG.cols*2/3, loIMG.rows*3/4);
//		loIMG = loIMG(a);
		Size dsize = Size(28, 28);

		resize(loIMG, img, dsize);

		cv::threshold(img, img, getaveragegray(img), 255, cv::ThresholdTypes::THRESH_BINARY_INV);


		imshow("pic2", img);
	}

	//CHECK(!img.empty()) << "Unable to decode image " << file;

	std::vector<Prediction> predictions = Classify(img);

	/* Print the top N predictions. */
//	for (size_t i = 0; i < predictions.size(); ++i)
//	{
//		Prediction p = predictions[i];
//		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//				<< p.first << "\"" << std::endl;
//	}
	return predictions[0].first;
}

int testcaffe()
{
	string caffe_root = "/home/kohill/mnist/";
	string model_file = caffe_root + "lenet.prototxt";
	string trained_file = caffe_root + "lenet_iter_20000.caffemodel";
	string mean_file = caffe_root + "mean.binaryproto";
	string label_file = caffe_root + "mnistlabel.txt";
	Classifier classifier(model_file, trained_file, mean_file, label_file);

	static int count;
	count++;
	count=count<1?1:count;count=count>16?1:count;
	string NumString = static_cast<ostringstream*>( &(ostringstream() << count) )->str();
	string file = "/home/kohill/Desktop/caffe data/cl/1/i_img_write10544.png";

	cv::Mat IMG = cv::imread(file, 0);
	cv::Mat img = Mat(28,28,0);
	//imshow("pic1", IMG);

	cv::Rect a = cv::Rect(25, 7, 14, 12);


	if(!IMG.empty())
	{
		IMG = IMG(a);
		Size dsize = Size(28,28);
		resize(IMG,img,dsize);
		cv::threshold(img, img, 140, 255,cv::ThresholdTypes::THRESH_BINARY_INV);

		imshow("pic2", img);
	}
	cv::waitKey(5);

	  std::cout << "---------- Prediction for "
	            << file << " ----------" << std::endl;


	  CHECK(!img.empty()) << "Unable to decode image " << file;
	  std::vector<Prediction> predictions = classifier.Classify(img);

	  /* Print the top N predictions. */
	  for (size_t i = 0; i < predictions.size(); ++i) {
	    Prediction p = predictions[i];
	    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
	              << p.first << "\"" << std::endl;
	  }

}


