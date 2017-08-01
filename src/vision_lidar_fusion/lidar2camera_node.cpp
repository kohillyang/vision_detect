#include "vision_lidar_fusion/lidar2camera_node.h"
#include <cmath>

Lidar2Camera::Lidar2Camera()
{
  ros::NodeHandle n;
  ros::Rate loop(1);
  sub = n.subscribe<sensor_msgs::LaserScan>("scan", 10, boost::bind(&Lidar2Camera::transfData, this, _1));
  pub = n.advertise<sensor_msgs::LaserScan>("transformed_scan", 10);
  ros::spin();
}
void Lidar2Camera::transfData(const sensor_msgs::LaserScan::ConstPtr scan)
{
  scan_ = *scan;
  //There should run tf
  for(int i = 0; i<scan_.ranges.size(); i++)
  {
    double transformed_x = scan_.ranges[i]*std::sin(M_PI - i) + 0.3;
    double transformed_y = scan_.ranges[i]*std::cos(M_PI - i) + 0.01;
    double transformed_d = sqrt(transformed_x*transformed_x + transformed_y*transformed_y);
    double transformed_r = atan(transformed_y/transformed_x);
    double dist ;
    transformed_scan_.dist[i]  = transformed_d;
    transformed_scan_.angle[i] = transformed_r;
  }
  pub.publish(transformed_scan_);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar2camera");
  Lidar2Camera data_transf;
  return 0;
}
