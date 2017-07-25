#include "vision_param.h"

namespace autocar
{
namespace vision_mul
{
vision_param::vision_param()
{
  read_params();
}

void vision_param::read_params()
{
  //vision_mul
  cv::FileNode vision_mul = root["vision_mul"];
  vision_mul["camera_matrix"] >> camera_matrix;
  vision_mul["dist_coeffs"]   >> dist_coeffs;
}
}
}
