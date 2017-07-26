/*
 * circle_detect.cpp
 *
 *  Created on: Jul 25, 2017
 *      Author: kohill
 */
#include "armor_detect_node.h"
#include "debug_utility.hpp"
#include "util.h"
#include "draw.h"
#include "vision_unit/armor_msg.h"
#include "labeler.h"
#include "video_recoder.h"
#include <memory>
#include <math.h>

namespace autocar
{
namespace vision_mul
{

float point_distance(const cv::Point2f point,const cv::Point2f center){
	auto dx = point.x - center.x;
	auto dy = point.y - center.y;
	return std::sqrt(dx*dx+dy*dy);
}
float point_circle_distance(const cv::Point2f point,const cv::Point2f center,float radius){
	auto x0 = point.x;
	auto y0 = point.y;
	float dis = point_distance(point,center);
	if(dis <= radius){
		return radius -dis ;
	}else{
		return radius + dis;
	}
}

#include "kohill.h"
bool isCircle(const std::vector<cv::Point> &points,const cv::Point &center,float radius,float &return_dis){
	if(radius > 3.5){
		const int perDegree = 12;
		std::vector<bool> flag(360/perDegree);//default false.
		for(auto iter = flag.begin();iter != flag.end();iter ++){
			*iter = false;
		}

		for(int i = 0;i<points.size();i++){
			const float theta = std::atan2(points[i].x - center.x,points[i].y - center.y) *180/3.14f + 180;
			int z = (int)(theta/perDegree);
			kohill::print("ltheta[i]",z);
			flag[z] = true;
		}
		bool isClose = true;
		for(int i = 0;i<flag.size();i++){
			isClose = isClose && flag[i];
		}
		bool r = false;
		if(isClose){
			float dis_points_ciecle = 0;
			for(int i = 0;i<points.size();i++){
				dis_points_ciecle += point_circle_distance(points[i],center,radius);
			}
			if(dis_points_ciecle < 6.28*radius*10){
				return_dis = dis_points_ciecle - 6.28*radius*10;
				return  dis_points_ciecle < 6.28*radius*8;
			}
			else{
				return false;
			}
		}else{
			return false;
		}
	}
	return false;
}
cv::Point2f his0,his1,his2;
bool detectCircle(const cv::Mat &img,cv::Point2f &center,float &){
	cv::Mat m_gray,m_gray_binary,img_circle = img.clone();
    cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::GaussianBlur(m_gray, m_gray, cv::Size(7, 7), 2, 2);
    cv::Canny(m_gray,m_gray_binary,80,255);
    cv::imshow("Canny",m_gray_binary);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_gray_binary,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    float return_dis;
    float return_dis_min=0;
	float latest_radius = 0;
	cv::Point2f latest_center;
	for (size_t i = 0; i < contours.size() ; i++) {
		float radius = 0;
		cv::Point2f center;
		cv::minEnclosingCircle(cv::Mat(contours[i]), center, radius);
		if(isCircle(contours[i],center,radius,return_dis)){
			if(return_dis < return_dis_min){
				return_dis_min = return_dis;
				latest_center = center;
				latest_radius = radius;
			}
		}
		if(point_distance(his0,his1) <  15 && point_distance(his2,his1) <  15){
			cv::circle(img_circle, latest_center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
			cv::circle(img_circle, latest_center, latest_radius, cv::Scalar(255, 0, 0), 3,
						8, 0);
		}

	}
	his2 = his1;
	his1 = his0;
	his0 = latest_center;
	cv::imshow("img_circle",img_circle);
	return true;
}

bool detectRectangle(const cv::Mat &img){
	cv::Mat m_gray,m_binary_r_sub_b,m_binary_color;
	static cv::Mat r_his[3];
	cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	std::vector<cv::Mat> bgr_channel;
	cv::split(img, bgr_channel);

	r_his[2] = r_his[1];
	r_his[1] = r_his[0];
	r_his[0] = (bgr_channel[2]-bgr_channel[0]) * 0.4 + 0.33*r_his[1] + 0.33*r_his[2];

	cv::threshold(r_his[0], m_binary_r_sub_b, 80, 255, cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
    cv::threshold(m_gray, m_binary_color, 30, 255, cv::ThresholdTypes::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::dilate(m_binary_color, m_binary_color, element, cv::Point(-1, -1), 3);
    auto m_binary_light = m_binary_color & m_binary_r_sub_b; // 两二值图交集
    cv::imshow("m_binary_r_sub_b_&_hight",m_binary_light);

	cv::Mat m_gray_binary,img_circle = img.clone();
    cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::GaussianBlur(m_gray, m_gray, cv::Size(7, 7), 2, 2);
    cv::Canny(m_gray,m_gray_binary,200,255);
    cv::imshow("Canny_Rec",m_gray_binary);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_gray_binary,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
}



}
}





