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

const float RECT_MAX_LEAN =  15;//15degree

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
float rectlongLean(const cv::RotatedRect &rect,float &w,float &h){
	cv::Point2f vertex[4];
	rect.points(vertex);
	cv::Point2f vertex_orderd[4];
	w = point_distance(vertex[0],vertex[1]);
	h = point_distance(vertex[1],vertex[2]);
	float angle = 0;
	if(w > h){
		std::swap(w,h);
		angle = std::abs(std::abs(std::atan2(vertex[1].y-vertex[0].y,vertex[1].x-vertex[0].x)) * 180/3.141592653-90);
	}else{
		angle = std::abs(std::abs(std::atan2(vertex[1].y-vertex[2].y,vertex[1].x-vertex[2].x)) * 180/3.141592653 - 90);
	}
	return angle;
}

#include "kohill.h"
static void drawRect(cv::Mat &img,const cv::RotatedRect rect){
	for (int i=0; i<4; i++){
		cv::Point2f vertex[4];
		rect.points(vertex);
		cv::line(img, vertex[i], vertex[(i+1)%4],cv::Scalar(0,255,0),6 );
	}
	return ;
}
static void drawCircle(cv::Mat &img,cv::Point2f center,float radius){
	cv::circle(img, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
	cv::circle(img, center, radius, cv::Scalar(255, 0, 0), 3,
				8, 0);
}
bool isCircle(const std::vector<cv::Point> &points,const cv::Point &center,float radius){
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
			if(dis_points_ciecle < 6.28*radius*20){
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
bool detectCircles(const cv::Mat &img,std::vector<cv::Point2f> &centers ,std::vector<float> &radiuses){
	cv::Mat m_gray,m_gray_binary,img_circle = img.clone();
    cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::GaussianBlur(m_gray, m_gray, cv::Size(7, 7), 2, 2);
    cv::Canny(m_gray,m_gray_binary,80,128);
    cv::imshow("Canny",m_gray_binary);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_gray_binary,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours.size() ; i++) {
		float radius = 0;
		cv::Point2f center;
		cv::minEnclosingCircle(cv::Mat(contours[i]), center, radius);
		if(isCircle(contours[i],center,radius)){
			centers.push_back(center);
			radiuses.push_back(radius);
		}
	}
	return true;
}
bool detectRectangle(const cv::Mat &img,std::vector<cv::RotatedRect> &rects){
	cv::Mat m_gray,m_binary_r_sub_b,m_binary_color;
	static cv::Mat r_his[3];
	cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	std::vector<cv::Mat> bgr_channel;
	cv::split(img, bgr_channel);

//	r_his[2] = r_his[1];
//	r_his[1] = r_his[0];
	m_binary_r_sub_b = (bgr_channel[2]-bgr_channel[0]);

	cv::threshold(m_binary_r_sub_b, m_binary_r_sub_b, 80, 255, cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
    cv::threshold(m_gray, m_binary_color, 30, 255, cv::ThresholdTypes::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::dilate(m_binary_color, m_binary_color, element, cv::Point(-1, -1), 3);
    auto m_binary_light = m_binary_color & m_binary_r_sub_b; // 两二值图交集
	cv::imshow("m_binary_r_sub_b_&_hight",m_binary_light);

    cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::GaussianBlur(m_gray, m_gray, cv::Size(7, 7), 2, 2);
    cv::Canny(m_gray,m_gray,88,220);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_gray,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    auto x_show_img = img.clone();
    for (auto iter = contours.begin();iter != contours.end();iter ++){
    	float lean_angle;
    	cv::RotatedRect rect = cv::minAreaRect(*iter);
		float w,h;
		float angle = rectlongLean(rect,w,h);
		if(angle < 15 && (h/w >1.5 && h/w < 16)){
			rects.push_back(rect);
		}
    }
    cv::imshow("Canny_Rec",x_show_img);
    return false;
}

void kohill_armor_detect(const cv::Mat &img){
	auto img_show = img.clone();
	std::vector<cv::RotatedRect> rects;
	std::vector<cv::Point2f> centers;
	std::vector<float> radiuse;
	detectRectangle(img,rects);
	detectCircles(img,centers,radiuse);
	std::vector<cv::Point2f > circle_armor_center;
	std::vector<float> circle_armor_radius;
	std::vector<cv::RotatedRect> rects_armor;

	for(int j=0; j < centers.size();j ++ ){
		std::vector<cv::RotatedRect> rects_temp;
		for(int i = 0; i < rects.size();i++){
			if(point_distance(centers[j],rects[i].center) < 64 ){
				rects_temp.push_back(rects[i]);
			}
		}
		if(rects_temp.size() == 2){
			for(int k = 0;k < rects_temp.size();k++ ){
				rects_armor.push_back(rects_temp[k]);
				circle_armor_center.push_back(centers[j]);
				circle_armor_radius.push_back(radiuse[j]);
			}

		}
	}
	for(int i = 0;i< rects_armor.size();i++){
		drawRect(img_show,rects_armor[i]);
	}
	for(int i = 0;i< circle_armor_center.size();i++){
		drawCircle(img_show,circle_armor_center[i],circle_armor_radius[i]);
	}
	auto x_all = img.clone();
	for(int i = 0;i< rects.size();i++){
		drawRect(x_all,rects[i]);
	}
	for(int i = 0;i< radiuse.size();i++){
		drawCircle(x_all,centers[i],radiuse[i]);
	}
	cv::imshow("img_show",img_show);
	cv::imshow("x_all",x_all);
}


}

}





