#ifndef ANLS_SCAN_H
#define ANLS_SCAN_H
#include <sensor_msgs/LaserScan.h>
#include <vector>
#include <ros/ros.h>
#include <vision_unit/transformed_scan.h>

class Lidar2Camera
{
public:
  Lidar2Camera();
  void transfData(const sensor_msgs::LaserScan::ConstPtr scan);
private:
  sensor_msgs::LaserScan scan_;
  vision_unit::transformed_scan transformed_scan_;
  ros::Subscriber sub;
  ros::Publisher pub;
};

#endif // ANLS_SCAN_H
