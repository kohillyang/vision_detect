#include "vision_lidar_fusion/vision_lidar_fusion.h"
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

using namespace std;
using namespace cv;

VisionLidarFusion::visionLidar_fusion()
{
  ros::NodeHandle n;
  ros::Rate loop(1);
  sub_transformed_scan_ = n.subscribe<vision_unit::transformed_scan>("transformed_scan", 10, boost::bind(&Lidar2Camera::transfData, this, _1));
  sub_transformed_scan_ = n.subscribe<vision_unit::armor_msg>("armor_info", 10, boost::bind(&Lidar2Camera::transfData, this, _1));
  ros::spin();
}
void VisionLidarFusion::fusionLidarCam(const vision_unit::transformed_scanConstPtr transformed_scan)
{
  for(int i=1;i<transformed_scan.angle.size();i++);
  {
    if(armor_angle_ > transformed_scan.angle[i-1] &&
       armor_angle_ < transformed_scan.angle[i])
      armor_index_ = i;
  }
  armor_dist_ = transformed_scan.dist[i];
  std::cout << armor_dist_ << std::endl;
}
void VisionLidarFusion::receive_armor_info(const vision_unit::armor_msgConstPtr armor_info)
{
  armor_angle_ = armor_info.x/640*(M_PI/4);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar2camera");
  VisionLidarFusion vision_lidar_fusion;
  return 0;
}
