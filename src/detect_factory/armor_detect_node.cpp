#include "armor_detect_node.h"
#include "debug_utility.hpp"
#include "util.h"
#include "draw.h"
#include "vision_unit/armor_msg.h"
#include "labeler.h"
#include "video_recoder.h"
#include <memory>
#include <math.h>
#include "circle_detect.hpp"
#define _DEBUG_VISION
namespace autocar
{
namespace vision_mul
{
armor_detect_node::armor_detect_node(void)
{

    ros::NodeHandle n;
    sub_yaw = n.subscribe<serial_comm::car_speed>("car_info", 5, boost::bind(&armor_detect_node::pan_info_callback, this, _1));
    pub_armor_pos = n.advertise<vision_unit::armor_msg>("armor_info", 1000);
    pub_goal = n.advertise<move_base_msgs::MoveBaseGoal>("camera_goal", 10);
    n.param<bool>("/armor_detect/debug_on", debug_on, false);
    std::cout<<"Debug_on:"<<debug_on<<std::endl;

    //Set the goal value to geometry_msgs::PoseStamped.
    goal_pose.target_pose.header.stamp    = ros::Time::now();
    goal_pose.target_pose.header.frame_id = "base_link";
    goal_pose.target_pose.pose.position.x = 0;
    goal_pose.target_pose.pose.position.y = 0;
    goal_pose.target_pose.pose.position.z = 0;


    goal_pose.target_pose.pose.orientation.w = 0;
    goal_pose.target_pose.pose.orientation.x = 0;
    goal_pose.target_pose.pose.orientation.y = 0;

    //init pnp param
    obj_p.push_back(cv::Point3f(0.0,   0.0,  0.0));
    obj_p.push_back(cv::Point3f(125,   0.0,  0.0));
    obj_p.push_back(cv::Point3f(125,   60,   0.0));
    obj_p.push_back(cv::Point3f(0.0,   60,   0.0));

    camera_matrix = cv::Mat(3, 3, CV_64F, cam);
    dist_coeffs   = cv::Mat(5, 1, CV_64F, dist_c);

    rvec = cv::Mat::ones(3,1,CV_64F);
    tvec = cv::Mat::ones(3,1,CV_64F);
    forward_back = true;
    //cv::Mat(obj_p).convertTo(obj_points, CV_32F);
}

armor_detect_node::~armor_detect_node(void)
{
}
static void drawRect(cv::Mat &img,const cv::RotatedRect rect){
	for (int i=0; i<4; i++){
		cv::Point2f vertex[4];
		rect.points(vertex);
		cv::line(img, vertex[i], vertex[(i+1)%4],cv::Scalar(0,255,0),6 );
	}
	return ;
}
static std::chrono::time_point<std::chrono::system_clock> time_his = std::chrono::system_clock::now();
void armor_detect_node::detect(const cv::Mat &image)
{
	this->debug_on = true;
    cv::Mat image;
	cv::VideoCapture capture_camera_forward("/home/kohill/vision_dataset/14.avi");
//    cv::VideoCapture capture_camera_forward(0);
//    capture_camera_forward.set(CV_CAP_PROP_FRAME_WIDTH, 640);
//    capture_camera_forward.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    if(!capture_camera_forward.isOpened())
    {
        std::cout<<"Cannot open the camera!"<<std::endl;
        return;
    }
    set_camera_exposure("/dev/video0", 50);
    //{
    //  while (true) {
    //    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    //  }
    //}

    std::shared_ptr<armor_detecter> armor_detector;
    std::shared_ptr<labeler> label;
    std::shared_ptr<video_recoder> recoder;

    int nframes = 0;
    bool detected = false;
    //filter
    int detected_count = 0;
    int undetected_count = 3;

    for (;;)
    {
    	auto speed_test_start_begin_time = std::chrono::system_clock::now();
        capture_camera_forward >> image;

        cv::Point2f circle_center;
        float cirlce_r;
        if(image.empty())
        {
            std::cout<<"Image has no data!"<<std::endl;
            continue;
        }else{
            cv::imshow("detect result",image);
        	cv::RotatedRect rect;
            if(kohill_armor_detect(image,rect)){
            	float dis = 0.1*std::sqrt(1/rect.size.area());
				armor_pos.detected = true;
				armor_pos.x = rect.center.x - 320;
				armor_pos.y = rect.center.y - 240;
				armor_pos.d = dis*1000;
                pub_armor_pos.publish(armor_pos);

                goal_pose.target_pose.pose.position.z = 0;

                goal_pose.target_pose.header.frame_id = "base_link";
                goal_pose.target_pose.header.stamp = ros::Time::now();
                goal_pose.target_pose.pose.position.x = 0;
                goal_pose.target_pose.pose.position.y = 0;
                goal_pose.target_pose.pose.position.z = dis;
                //tf::Quaternion q;
                //q.setRPY(0, 0, pan_yaw);
                goal_pose.target_pose.pose.orientation.x= 0;
                goal_pose.target_pose.pose.orientation.y= 0;
                goal_pose.target_pose.pose.orientation.z= 0;
                goal_pose.target_pose.pose.orientation.w= 1;
                pub_goal.publish(goal_pose);
                auto image_tmp = image.clone();
                drawRect(image_tmp,rect);
                cv::imshow("detect result",image_tmp);
            }else{
            	armor_pos.detected = false;
            	pub_armor_pos.publish(armor_pos);
            	goal_pose.target_pose.pose.position.z =0;
                pub_goal.publish(goal_pose);
            }
        	pub_armor_pos.publish(armor_pos);
        }
        cv::waitKey(1);
    }
}

void armor_detect_node::pan_info_callback(const serial_comm::car_speed::ConstPtr &pan_data)
{
    //pan_yaw = pan_data->yaw;
}

bool armor_detect_node::if_detected_armor()
{
    return detected_armor;
}

bool armor_detect_node::get_camera_num()
{
    return forward_back;
}

armor_info* armor_detect_node::get_armor()
{
    return armor_;
}

void armor_detect_node::set_image_points(std::vector<cv::Point2f> armor_points)
{
    //  cv::Point2f vertices[4];
    //  rect.points(vertices);
    //  cv::Point2f lu, ld, ru, rd;
    //  std::sort(vertices, vertices + 4, [](const cv::Point2f & p1, const cv::Point2f & p2) { return p1.x < p2.x; });
    //  if (vertices[0].y < vertices[1].y){
    //    lu = vertices[0];
    //    ld = vertices[1];
    //  }
    //  else{
    //    lu = vertices[1];
    //    ld = vertices[0];
    //  }
    //  if (vertices[2].y < vertices[3].y)	{
    //    ru = vertices[2];
    //    rd = vertices[3];
    //  }
    //  else {
    //    ru = vertices[3];
    //    rd = vertices[2];
    //  }
    //  //img_p.push_back()
    //  img_p.clear();
    //  img_p.push_back(lu);
    //  img_p.push_back(ru);
    //  img_p.push_back(rd);
    //  img_p.push_back(ld);
    cv::Mat(armor_points).convertTo(img_points, CV_32F);
}
}
}

