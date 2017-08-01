#ifndef DETECT_FACTORY_H
#define DETECT_FACTORY_H
#include <opencv2/opencv.hpp>
namespace autocar
{
namespace vision_mul
{
class detect_factory
{
public:
  virtual bool detect(const cv::Mat &image, bool detect_blue) = 0;
  //virtual void get_info() = 0;
};
}
}

#endif // DETECT_FACTORY_H
