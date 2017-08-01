#ifndef __ARMOR_H__
#define __ARMOR_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <tuple>
#include <chrono>
#include "detect_factory.h"

namespace autocar
{
namespace vision_mul
{
class armor_info
{
public:
  armor_info(int type_, cv::RotatedRect armor_, std::vector<cv::Point2f> points_, cv::RotatedRect left_=cv::RotatedRect(), cv::RotatedRect right_=cv::RotatedRect())
  {
    type = type_;
    armor = armor_;
    points = points_;
    left_light = left_;
    right_light = right_;
    score = 0.0f;
  }
  armor_info();

  int type;
  cv::RotatedRect armor;
  std::vector<cv::Point2f> points;
  cv::RotatedRect left_light;
  cv::RotatedRect right_light;
  float score;
};

class armor_detecter: public detect_factory
{
private:

  const static float m_threshold_max_angle;
  const static float m_threshold_min_area;
  const static float m_threshold_dis_lower;
  const static float m_threshold_dis_upper;
  const static float m_threshold_len;
  const static float m_threshold_height;
  const static float m_threshold_gray;

public:

  /**
   * @brief constructor
   */
  armor_detecter(bool debug_on);
  /**
   * @brief Highlight the blue or red region of the image
   * @param image input image ref
   * @param detect_blue true represent blue, false represent red
   * @return gray image
   */
  cv::Mat highlight_blue_or_red(const cv::Mat &image, bool detect_blue);
  /**
   * @brief The wrapper for function cv::findContours
   * @param binary binary image
   * @return contour
   */
  std::vector<std::vector<cv::Point>> find_contours(const cv::Mat &binary);
  /**
   * @brief Finding the true light contour
   * @param contours_light blue or red region contour
   * @return light contour
   */
  std::vector<cv::RotatedRect> point_to_rects(const std::vector<std::vector<cv::Point>> &contours);
  /**
   * @brief Finding the true light contour
   * @param contours_light blue or red region contour
   * @param contours_brightness gray image contour
   * @return light contour
   */
  std::vector<cv::RotatedRect> to_light_rects(const std::vector<std::vector<cv::Point>> &contours_light, const std::vector<std::vector<cv::Point>> &contours_brightness);
  /**
   * @brief detect the lights on the armors
   * @param detect_blue true represent blue, false represent red
   * @return
   */
  std::vector<cv::RotatedRect> detect_lights(bool detect_blue);
  /**
   * @brief Filtering the detected lights
   * @param lights detected lights ref
   * @param thresh_max_angle the max area of the lights
   * @param thresh_min_area the min area of the lights
   * @return
   */
  std::vector<cv::RotatedRect> filter_lights(const std::vector<cv::RotatedRect> &lights, float thresh_max_angle, float thresh_min_area);
  /**
   * @brief The possible armors
   * @param detected lights beside the armors
   * @param thres_max_angle the max angle of the armor
   * @param thres_dis_lower
   * @param thres_dis_upper
   * @param thres_len
   * @param thres_height
   * @param thres_gray
   * @param detect_blue
   * @return
   */
  std::vector<armor_info> possible_armors(const std::vector<cv::RotatedRect> &lights, float thres_max_angle, float thres_dis_lower, \
                                                float thres_dis_upper, float thres_len, float thres_height, float thres_gray, bool detect_blue);
  void cal_armor_info(std::vector<cv::Point2f> &armor_points, cv::RotatedRect left_light, cv::RotatedRect right_light);
  /**
   * @brief Filtering Detected armors by aspect ratio, conv and so on
   * @param armors detected armors ref
   * @return filtered armors
   */
  std::vector<armor_info> filter_by_features(std::vector<armor_info> &armors);
  /**
   * @brief The entrance function of armor detect
   * @param image rgb image
   * @return return ture if find armor, else return false
   */
  bool detect(const cv::Mat &image, bool detect_blue);
  /**
   * @brief Slect one armor as the target armor which we will shoot
   * @param all_armors all detected armors
   * @return the target armor
   */
  void slect_final_armor(std::vector<armor_info> all_armors);
  /**
   * @brief Get the private armor information
   * @return
   */
  armor_info* get_armor();
  /**
   * @brief Show the processed image for debug
   */
  void debug_vision();

private:

  cv::Mat m_common;
  cv::Mat m_image;
  cv::Mat m_image_hql;
  cv::Mat m_show;
  cv::Mat m_gray;
  cv::Mat possible_armor;
  cv::Mat light_img1;
  cv::Mat light_img2;
  cv::Mat light_img;
  cv::Mat m_binary_brightness;
  cv::Mat m_binary_color;
  cv::Mat m_binary_light;
  bool debug_on_;

  std::chrono::system_clock::time_point speed_test_start_begin_time;
  armor_info *armor;
  armor_info *old_armor;
};
}
}


#endif // !__ARMOR_H__
