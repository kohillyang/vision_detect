#include <ros/ros.h>
#include <vision_unit/SetGoal.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "draw.h"
#include "debug_utility.hpp"
#include "armor_detect.h"
#include "armor_detect_node.h"
#include <geometry_msgs/PoseStamped.h>
#include <vision_unit/armor_msg.h>

int main(int argc, char **argv)
{

  ros::init(argc, argv, "armor_detect");
  try
  {
    //Try to detect armor.
    static autocar::vision_mul::armor_detect_node armor_solver;
    armor_solver.running();
  }
  catch (const std::exception &ex)
  {
    std::cout << ex.what() << std::endl;
  }
  return 0;
}

