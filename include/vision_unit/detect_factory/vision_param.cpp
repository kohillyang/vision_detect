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
  cv::FileStorage fs("/home/dji/dji_beta/auto_car/src/param_manager/autocar_params.xml", cv::FileStorage::READ);
  if(!fs.isOpened())
  {
    std::cout << "Cannot open aruocar_params.xml, please check if the file is exist." << std::endl;
  }
  root = fs.root();
  //vision_mul
  cv::FileNode vision_mul = root["vision_mul"];
  vision_mul["camera_matrix"] >> camera_matrix;
  vision_mul["dist_coeffs"]   >> dist_coeffs;
  //std::cout << camera_matrix.size() << std::endl << camera_matrix << std::endl;
  //std::cout << dist_coeffs.size() << std::endl << dist_coeffs << std::endl;
}
}
}
