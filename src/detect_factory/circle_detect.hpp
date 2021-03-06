/*
 * circle_dect.hpp
 *
 *  Created on: Jul 26, 2017
 *      Author: kohill
 */

#ifndef SRC_DETECT_FACTORY_CIRCLE_DETECT_HPP_
#define SRC_DETECT_FACTORY_CIRCLE_DETECT_HPP_
#include <opencv2/opencv.hpp>
namespace autocar
{
namespace vision_mul
{
bool kohill_armor_detect(const cv::Mat &img,cv::RotatedRect &rect);
}
}


#endif /* SRC_DETECT_FACTORY_CIRCLE_DETECT_HPP_ */
