#ifndef _UTIL_H_
#define _UTIL_H_

#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>
#include <string>

namespace autocar
{
namespace vision_mul
{
    // 通过v4l2配置摄像头参数
    int set_camera_exposure(std::string id, int val);
}
}

#endif // !_UTIL_H_
