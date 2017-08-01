#ifndef __THREAD_OPERATION_H__
#define __THREAD_OPERATION_H__
//
#include <ros/ros.h>
#include<tf/tf.h>
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <opencv2/opencv.hpp>
#include "armor_detect.h"
#include <logical_core/SetGoal.h>
#include <vision_unit/armor_msg.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <serial_comm/car_speed.h>
#include "armor_detect.h"
#include <cmath>

namespace autocar
{
namespace vision_mul
{
class armor_detect_node
{
public:
    armor_detect_node(void);
    ~armor_detect_node(void);

    /**
     * @brief running the program of armor detect
     */
    void running(void);
    /**
     * @brief If the program can detect an armor
     * @return ture if detected an armor
     */
    bool if_detected_armor();
    void pan_info_callback(const serial_comm::car_speed::ConstPtr &pan_data);
    /**
     * @brief Maybe we have more than one camera, so we can use this function to get the camera No. that detected armor
     * @return
     */
    bool get_camera_num();
    /**
     * @brief Obtaining the armor information including the width and hight of the armor
     * @return
     */
    armor_info* get_armor();

private:
    void set_image_points(std::vector<cv::Point2f> armor_points);

private:
    //vision_param *param;
    bool detected_armor;
    armor_info *armor_;
    float pan_yaw;

    cv::Mat m_img_buff[2];
    bool forward_back;
    bool debug_on;
    ros::Subscriber sub_yaw;
    ros::Publisher pub_armor_pos, pub_goal;
    vision_unit::armor_msg armor_pos;
    move_base_msgs::MoveBaseGoal goal;
    move_base_msgs::MoveBaseGoal goal_pose;

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Mat obj_points;
    cv::Mat img_points;
    std::vector<cv::Point3f> obj_p;
    std::vector<cv::Point2f> img_p;
    cv::Mat rvec;
    cv::Mat tvec;
    double ang_lim = 3.0/180*3.14;
    double cam[9] = {839.923052, 0.0,        340.780730,
                     0.0,        837.671081, 261.766523,
                     0.0,        0.0,        1.0       };

    double dist_c[5] = {0.082613, 0.043275, 0.002486, -0.000823, 0.0};
};
}
}

#endif // !__THREAD_OPERATION_H__

