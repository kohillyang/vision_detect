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
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <tf/tf.h>
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

#define USB_USB_CAM 0
#if USB_USB_CAM
namespace autocar
{
namespace vision_mul
{
class armor_detect;
}
}


static const std::string OPENCV_WINDOW = "Image window";
using autocar::vision_mul::armor_detect;



class ImageConverter {
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	static void imageCb(const sensor_msgs::ImageConstPtr &msg){
		std::cout << "hello world" << std::endl;
		static autocar::vision_mul::armor_detect_node *armor_solver = 0;
		if(!armor_solver){
			armor_solver = new autocar::vision_mul::armor_detect_node;
		}
		cv_bridge::CvImagePtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvCopy(msg,
					sensor_msgs::image_encodings::BGR8);
		} catch (cv_bridge::Exception &e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		cv::imshow(OPENCV_WINDOW, cv_ptr->image);
		armor_solver->detect(cv_ptr->image);
	}
public:
	ImageConverter() :
			it_(nh_) {
		std::cout << "hello world" << std::endl;
		image_sub_ = this->it_.subscribe("/usb_cam/image_raw", 1, ImageConverter::imageCb);
		cv::namedWindow(OPENCV_WINDOW);

	}

	~ImageConverter() {
		cv::destroyWindow(OPENCV_WINDOW);
	}
};

int main(int argc, char **argv) {

	ros::init(argc, argv, "armor_detect");
	volatile ImageConverter ic = ImageConverter();
	ros::spin();
	return 0;
}
#else
int main(int argc, char **argv) {
	ros::init(argc, argv, "armor_detect");
	autocar::vision_mul::armor_detect_node armor_solver;
    cv::Mat image;
	cv::VideoCapture capture_camera_forward("/home/kohill/vision_dataset/13.avi");
//	    cv::VideoCapture capture_camera_forward(0);
    capture_camera_forward.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    capture_camera_forward.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	for(;;){
	    capture_camera_forward >> image;
	    //cv::imshow("im",image);
	    cv::waitKey(1);
	    armor_solver.detect(image);
	}
	ros::spin();
	return 0;
}
#endif

