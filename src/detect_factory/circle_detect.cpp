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
    PyObject *func_tryFindMinAreaRect;
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
        cout << "start loading Python Module..." << endl;
        Py_Initialize();
        PyRun_SimpleString("import sys,os");
        PyRun_SimpleString((string("sys.path.append('")+getResAbsolutePath()+"')").data());
        PyRun_SimpleString("import trainAndTest");
        PyObject* moduleName = PyString_FromString("AreaRect");
        PyObject* pModule = PyImport_Import(moduleName);
        if (!pModule) // 加载模块失败
        {
            cerr << "[ERROR] Python get module failed." << endl;
            exit(-1);
        }
        func_tryFindMinAreaRect = PyObject_GetAttrString(pModule, "tryFindMinAreaRect");
        if (!func_tryFindMinAreaRect || !PyCallable_Check(func_tryFindMinAreaRect))
        {
            cerr << "[ERROR] Can't find function (tryFindMinAreaRect)" << endl;
            exit(-1);
        }
        cout << "[info] Python Module load Finished." << endl;

    }
    bool tryFindMinAreaRect(const cv::Mat &img,cv::Mat &img_outPut){




        cv::Mat m_gray;
        cv::cvtColor(img, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
        cv::imwrite("/tmp/armor_detect_tmp.png",img);
        cv::resize(m_gray, m_gray, cv::Size(64, 32), (0, 0), (0, 0), cv::INTER_CUBIC);
        int img_size = m_gray.cols * m_gray.channels() *m_gray.rows;
        PyObject* args = PyTuple_New(1);
        PyObject *pListArg1 = PyList_New(0);
        for (int i = 0;i<img_size;i++){
        	PyObject *p = Py_BuildValue("b",m_gray.data[i]);
        	PyList_Append(pListArg1,p);
        }
        PyTuple_SetItem(args,0,pListArg1);
        PyObject *r_object = PyObject_CallObject(func_tryFindMinAreaRect, args);
        int detected=0,x,y,w,h;
        PyArg_ParseTuple(r_object,"i|i|i|i|i",&detected,&x,&y,&w,&h);
        if(detected == 1){
        	img_outPut = img(cv::Rect(x,y,w,h));
        	return true;
        }else{
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
void deltaAndMean(const std::vector<float> &ve,float &mean,float &delta){
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

	deltaAndMean(col_sum,mean_x,delta_x);
	deltaAndMean(row_sum,mean_y,delta_y);
	kohill::print("\nmean Value\t\t\t\t\t\t",delta_x," ",delta_y);
	rect  = cv::RotatedRect(
			cv::Point2f(mean_x,mean_y),
			cv::Size2f(64,32),0);
	return false;
}

rectang_detecter recDetector(false);
//Classifier calssifier;
bool kohill_armor_detect(const cv::Mat &img,cv::RotatedRect &rect_out){
	auto img_show = img.clone();

	std::vector<cv::Point2f> centers;
	std::vector<float> radiuse;

	std::vector<rectangdetect_info> rectangs = recDetector.detect_enemy(img,true);
	auto img_lights = recDetector.m_binary_light;
	cv::RotatedRect car_rect;
	autocar::vision_mul::kohill_car_detect(img_lights,car_rect);
	auto img_car = img.clone();
	drawRect(img_car,car_rect,cv::Scalar(255,255,255));

	std::vector<cv::RotatedRect> rects_last;
	std::vector<float> delta_last;
	for (auto x = rectangs.begin(); x != rectangs.end(); x++)
	{
		auto rect = x->rect.boundingRect();
		if (rect.br().x < img.size().width && rect.br().y < img.size().height
				&& rect.tl().x > 0 && rect.tl().y > 0)
		{
			auto image_sub_sub = img(rect);
			cv::Mat img_out;

			if (pythonCircleDetector.tryFindMinAreaRect(image_sub_sub,img_out))
			{
				rects_last.push_back(x->rect);
				delta_last.push_back(0);
				cv::imshow("img_circle",img_out);
//				string str= calssifier.classify_mnist(img_out);
//				cout << "[info]: detect result"<<str << std::endl ;
			}
			else
			{
			}
		}
		else
		{
			cerr << "over flow.." << endl;
		}
	}
	kohill::kimshow("my img_car",img_car);

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
			return true;
		}
		return false;
	}
	return false;
}



}

}





