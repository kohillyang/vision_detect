/*
 * circle_detect.cpp
 *
 *  Created on: Jul 25, 2017
 *      Author: kohill
 */
#define CPU_ONLY 1
#include "armor_detect_node.h"
#include "debug_utility.hpp"
#include "util.h"
#include "draw.h"
#include "vision_unit/armor_msg.h"
#include "labeler.h"
#include "video_recoder.h"
#include <memory>
#include <math.h>
#include "detect_HQG.hpp"
#include "kohill.h"
#include <stdio.h>
#include <iostream>
#include "digital_classification.hpp"
using namespace std;
using namespace cv;
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Python.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
class PyCircleDetect{
private:
	std::vector<int> x2={
#include"txt/x2.txt"
	};
	std::vector<int> y2={
#include"txt/y2.txt"
	};
	std::vector<int> x3={
#include"txt/x3.txt"
	};
	std::vector<int> y3={
#include"txt/y3.txt"
	};
	std::vector<int> x4={
#include"txt/x4.txt"
	};
	std::vector<int> y4={
#include"txt/y4.txt"
	};
	std::vector<int> x5={
#include"txt/x5.txt"
	};
	std::vector<int> y5={
#include"txt/y5.txt"
	};

public:
    static std::string getResAbsolutePath(){
    	char currentFileName[]=__FILE__;
    	int fileNameLen = strlen(currentFileName);
    	do{
    		if(currentFileName[fileNameLen] == '/' ||
    				currentFileName[fileNameLen] == '\\' ){
    			break;
    		}else{currentFileName[fileNameLen]=0;}
    	}while(fileNameLen--);
    	string resFilePath = string(currentFileName) + "detect_res/";
    	return resFilePath;
    }

    PyCircleDetect(){

    }
    void xy2rad(){

    }
    bool tryFindMinAreaRect(const cv::Mat &img,cv::Mat &img_outPut,float &lsum_average){
    	if(img.rows < 5 ||  img.cols <10 || img.cols/img.rows < 1.3){
    		return false;
    	}
    	cv::Mat img_stdsize;
    	cv::resize(img, img_stdsize, cv::Size(64, 32), (0, 0), (0, 0), cv::INTER_CUBIC);
    	vector<Mat> bgr;
    	cv::split(img_stdsize(cv::Rect(15,0,32,32)),bgr);
    	int i = 0,sum_p = 0;
    	for(auto x = bgr[0].data;i<64*32;i++){
    		sum_p += x[i];
    	}
    	sum_p /= 32 *64;
    	if(sum_p < 40){
    		return false;
    	}
    	lsum_average = sum_p;
    	kohill::print("sum_mean_cut:",sum_p);
    	cv::Mat img_binary;
    	cv::threshold(bgr[0],img_binary,sum_p,255,cv::ThresholdTypes::THRESH_BINARY);
    	auto imgfunc = [&img_binary](int i, int j) { return *(img_binary.ptr<uchar>(i) + j); };
    	int r3_count = 0,r4_count = 0,r5_count=0;
    	for(int i = 0;i<x3.size();i++){
//    		kohill::print(x3[i],y3[i],int(imgfunc(x3[i],y3[i])));
    		r3_count += imgfunc(x3[i],y3[i]) > 1?1:0;
    	}
    	for(int i = 0;i<x4.size();i++){
    		r4_count += imgfunc(x4[i],y4[i]) > 1?1:0;
    	}
    	for(int i = 0;i<x5.size();i++){
    		r5_count += imgfunc(x5[i],y5[i]) > 1?1:0;
    	}
    	int  judge = 0 ;
    	if(r3_count >= 17){
    		judge += 1;
    	}
    	if(r4_count >= 10){
    		judge += 1;
    	}
    	if(r5_count >= 27){
    		judge += 1;
    	}

    	cv::imshow("input",img_stdsize);
    	cv::imshow("img_cut_binary",bgr[2]);
    	cv::imshow("input_img_binary",img_binary);

    	if(judge >=2){
        	kohill::print("yes",r3_count,r4_count,r5_count);
        	return true;
    	}else{
        	kohill::print("no ",r3_count,r4_count,r5_count);
        	kohill::print("size",x3.size(),x4.size(),x5.size());
        	return false;
    	}
    }
    ~PyCircleDetect(){
//        Py_Finalize();
    }
};
static PyCircleDetect pythonCircleDetector;
namespace autocar
{
namespace vision_mul
{

const float RECT_MAX_LEAN =  15;//15degree
const int img_width = 640;
const int img_height = 480;
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
static void drawRect(cv::Mat &img,const cv::RotatedRect rect,cv::Scalar color = cv::Scalar(0,255,0)){
	for (int i=0; i<4; i++){
		cv::Point2f vertex[4];
		rect.points(vertex);
		cv::line(img, vertex[i], vertex[(i+1)%4],color,6 );
	}
	return ;
}
static void drawCircle(cv::Mat &img,cv::Point2f center,float radius){
//	cv::circle(img, center, 0, cv::Scalar(0, 255, 0), -1, 8, 0);
	cv::circle(img, center, radius, cv::Scalar(255, 0, 0), 3,
				3, 0);
}
bool isCircle(const std::vector<cv::Point> &points,const cv::Point &center,float radius,float &delta){
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
				delta =dis_points_ciecle;
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
float detectCircles(const cv::Mat &img,std::vector<cv::Point2f> &centers ,std::vector<float> &radiuses){
	cv::Mat m_gray,m_gray_binary,img_circle = img.clone();
    cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
//    cv::GaussianBlur(m_gray, m_gray, cv::Size(7, 7), 2, 2);
    cv::Canny(m_gray,m_gray,80,128);
//	cv::threshold(m_gray, m_gray, 25, 255, cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_gray,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	float delta = 100000000;
	for (size_t i = 0; i < contours.size() ; i++) {
		float radius = 0;
		cv::Point2f center;
		cv::minEnclosingCircle(cv::Mat(contours[i]), center, radius);

		if(isCircle(contours[i],center,radius,delta)){
			centers.push_back(center);
			radiuses.push_back(radius);
		}
	}
	return delta;
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
    return false;
}

//@param img:light red & light red
void deltaAndMean(const std::vector<float> &ve,float &mean,float &delta,float &sumAll){
	float su = 0,delta_sum = 0,sum2 = 0;
	for(int i=0;i<ve.size();i++){
		sum2 += ve[i];
		su += i * ve[i];
	}
	mean = su/sum2;
	for(int i=0;i<ve.size();i++){
		float d = ve[i]*(i - mean)/sum2;
		delta_sum +=  d *d;
	}
	delta = std::sqrt(delta_sum);
	sumAll = sum2;
}
bool kohill_car_detect(const cv::Mat &img,cv::RotatedRect &rect){
	int  width = img.cols;
	int height = img.rows;
	auto imgfunc = [&img,&width](int i, int j) { return *(img.ptr<uchar>(i) + j); };
	float mean_x,mean_y,delta_x,delta_y;
	std::vector<float> col_sum,row_sum;
	for(int j = 0;j<width;j++ ){
		float s = 0;
		for(int i = 0;i<height;i++){
			s += imgfunc(i,j);
		}
		col_sum.push_back(s/height);
	}
	for(int i = 0;i<height;i++ ){
		float s = 0;
		for(int j = 0;j<width;j++){
			s += imgfunc(i,j);
		}
		row_sum.push_back(s/width);
	}
	float sumAll;
	deltaAndMean(col_sum,mean_x,delta_x,sumAll);
	deltaAndMean(row_sum,mean_y,delta_y,sumAll);
	cerr << ("[Warn]:no car detected,try using mean value instead.") << endl;
	std::cout << sumAll << std::endl;

	if(sumAll > 100){
		kohill::print("far cat detected.",delta_x," ",delta_y);
		rect  = cv::RotatedRect(
				cv::Point2f(mean_x,mean_y),
				cv::Size2f(6,3),0);
		return true;
	}
	return false;
}

rectang_detecter recDetector(false);
//Classifier calssifier;
bool kohill_armor_detect(const cv::Mat &img,cv::RotatedRect &rect_out){
	static Point2f lastResult(0,0);
	auto img_temp = img.clone();
	std::vector<cv::Point2f> centers;
	std::vector<float> radiuse;
	std::vector<rectangdetect_info> rectangs = recDetector.detect_enemy(img,false);
	auto img_car = img.clone();
	std::vector<cv::RotatedRect> rects_last;
	std::vector<float> delta_last;
	for (auto x = rectangs.begin(); x != rectangs.end(); x++)
	{
		auto rect = x->rect.boundingRect();
		if (rect.br().x < img.size().width && rect.br().y < img.size().height
				&& rect.tl().x > 0 && rect.tl().y > 0)
		{
			auto image_sub_sub = img_temp(rect);
			cv::Mat img_out;
			float l_sum;
			if (pythonCircleDetector.tryFindMinAreaRect(image_sub_sub,img_out,l_sum))
			{
				rects_last.push_back(x->rect);
				delta_last.push_back(point_distance(x->rect.center,lastResult));
				rect_out =x->rect;
			}
		}
		else
		{
			cerr << "over flow.." << endl;
		}
	}
	for(int i = 0;i<rects_last.size();i++){
		for(int j=0;rects_last.size()>1 && j<rects_last.size()-1;j++){
			if(delta_last[j] >delta_last[j+1] ){
				std::swap(delta_last[j],delta_last[j+1]);
				std::swap(rects_last[j],rects_last[j+1]);
			}
		}
	}
	if(rects_last.size()> 0 ){
		rect_out = rects_last[0];
		lastResult =rect_out.center;
		return true;
	}else{
		float x_center =0.0;
		float y_center =0.0;
		using autocar::vision_mul::rectangdetect_info;
		std::sort(rectangs.begin(),rectangs.end(),
				[](const autocar::vision_mul::rectangdetect_info &p1,
						const autocar::vision_mul::rectangdetect_info &p2)
				{return p1.lost < p2.lost;});
		if(rectangs.size()>0){
			rect_out =rectangs[0].rect;
			lastResult = rect_out.center;
			return true;
		}
		auto img_lights = recDetector.m_binary_light;
		cv::RotatedRect car_rect;
		if(autocar::vision_mul::kohill_car_detect(img_lights,car_rect)){
			rect_out=  car_rect;
			lastResult = rect_out.center;
			return true;
		}else{
			return false;
		}
	}
}



}

}





