 /* circle_dect.hpp
 *
 *  Created on: Jul 26, 2017
 *      Author: kohill
 */

#ifndef DETECT_HQG_HPP_
#define DETECT_HQG_HPP_

#include <opencv2/opencv.hpp>
#include "detect_factory.h"
namespace autocar
{
namespace vision_mul
{
class rectandetect_info
{
public:
	rectandetect_info(cv::RotatedRect rect_, std::vector<cv::Point2f> points_, cv::RotatedRect left_=cv::RotatedRect(), cv::RotatedRect right_=cv::RotatedRect(), float lost_ = 10000.0)
  {
    type = true;
    rect = rect_;
    points = points_;
    left_light = left_;
    right_light = right_;
    score = 0.0f;
    lost = lost_;
  }
	rectandetect_info();
  bool type;
  cv::RotatedRect rect;
  std::vector<cv::Point2f> points;
  cv::RotatedRect left_light;
  cv::RotatedRect right_light;
  float score;
  float lost;
};


class rectang_detecter:public detect_factory
{
private:
	cv::Mat m_common;
	cv::Mat m_image;
	cv::Mat m_image_hql;
	cv::Mat m_show;
	cv::Mat m_gray;
	cv::Mat final_rectang;
	cv::Mat possible_rectang;
	cv::Mat light_img;
	cv::Mat m_binary_brightness;
	cv::Mat m_binary_color;
	cv::Mat m_binary_light;

	const static float m_threshold_max_angle;
	const static float m_threshold_min_area;
	const static float m_threshold_max_area;

	const static float threshold_line_binary;
	const static float threshold_line_binary_color;

	bool debug_on_;

	rectandetect_info *finalrect;
	rectandetect_info *old_finalrect;

public:
	rectang_detecter(bool debug_on);

	std::vector<std::vector<cv::Point>> find_contours(const cv::Mat &binary);

	cv::Mat highlight_blue_or_red(const cv::Mat &image, bool detect_blue);

	double adjust_threshold_binary(const cv::Mat &image,double threshold, double threshold_line);

	std::vector<cv::RotatedRect> point_to_rects(const std::vector<std::vector<cv::Point>> &contours);

	std::vector<cv::RotatedRect> to_light_rects(const std::vector<std::vector<cv::Point>> &contours_light, const std::vector<std::vector<cv::Point>> &contours_brightness);

	std::vector<cv::RotatedRect>  detect_lights(bool detect_blue);

	float point_distance(const cv::Point2f point,const cv::Point2f center);

	float rectlongLean(const cv::RotatedRect &rect,float &w,float &h);

	rectandetect_info* select_final_rectang(std::vector<rectandetect_info>& rectangs);

	std::vector<rectandetect_info> detect_select_rect(const std::vector<cv::RotatedRect> &lights);

	std::vector<cv::RotatedRect> filter_lights(const std::vector<cv::RotatedRect> &lights, float thresh_max_angle, float thresh_min_area, float thresh_max_area);

	std::vector<cv::RotatedRect> detect_enemy_light(const cv::Mat &image, bool detect_blue);

	bool detect(const cv::Mat &image, bool detect_blue);
};
}
}

#endif /* SRC_DETECT_FACTORY_CIRCLE_DETECT_HPP_ */
