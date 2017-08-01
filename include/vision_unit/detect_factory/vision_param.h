#ifndef VISION_PARAM_H
#define VISION_PARAM_H
#include "/home/dji/dji_beta/auto_car/src/param_manager/include/param_manager.h"

namespace autocar
{
namespace vision_mul
{
class vision_param: public param_mul::param_manager
{
public:
  vision_param();
  /**
   * @brief Read vision paramaters, such as camera matrix, Distortion matrix and so on.
   */
  void read_params();
public:
  //vision_mul
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
};
}
}

#endif // VISION_PARAM_H
