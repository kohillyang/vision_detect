#ifndef VISION_LIDAR_FUSION_H
#define VISION_LIDAR_FUSION_H
#include "vision_lidar_fusion/vision_lidar_fusion.h"
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <vision_unit/transformed_scan.h>
#include <vision_unit/armor_msg.h>
struct RelPos
{
  double x;
  double y;
  double z;
};

struct AbsPos
{
  double x;
  double y;
  double z;
};

class VisionLidarFusion
{
public:
  VisionLidarFusion();
  void fusionLidarCam(const vision_unit::transformed_scanConstPtr transformed_scan);
  void receive_armor_info(const vision_unit::armor_msgConstPtr transformed_scan);
private:
  ros::Subscriber sub_camera_;
  ros::Subscriber sub_transformed_scan_;
  double armor_angle_;
  int armor_index_;
  double armor_dist_;
};

#endif // VISION_LIDAR_FUSION_H
