/*
 * regt_dte_HQG.cpp
 *
 *  Created on: Jul 26, 2017
 *      Author: HQG
 */
#include "armor_detect_node.h"
#include "debug_utility.hpp"
#include "circle_detect.hpp"
#include "util.h"
#include "draw.h"
#include "vision_unit/armor_msg.h"
#include "labeler.h"
#include "video_recoder.h"
#include "detect_HQG.hpp"
#include <memory>
#include <math.h>
#include <algorithm>

namespace autocar
{
namespace vision_mul
{
const float rectang_detecter::m_threshold_max_angle = 30.0f;

const float rectang_detecter::m_threshold_min_area = 3.0f;

const float rectang_detecter::m_threshold_max_area = 6000.0f;

const float rectang_detecter::threshold_line_binary = 8000.f;

const float rectang_detecter::threshold_line_binary_color = 4500.f;

rectandetect_info::rectandetect_info()
{
	type = false;
	rect = cv::RotatedRect();
	left_light = cv::RotatedRect();
	right_light = cv::RotatedRect();
	score = 0.0f;
	lost = 10000.0;
}
rectang_detecter::rectang_detecter(bool debug_on)
{
	finalrect = new rectandetect_info();
    debug_on_ = debug_on;
    old_finalrect = new rectandetect_info();
    old_finalrect->rect.center.x = 0;
    old_finalrect->rect.size.width = 0;
    old_finalrect->rect.size.height = 0;
}
std::vector<std::vector<cv::Point>> rectang_detecter::find_contours(const cv::Mat &binary)
{
    std::vector<std::vector<cv::Point>> contours; // 边缘
    const auto mode = cv::RetrievalModes::RETR_LIST;//RETR_EXTERNAL;
    const auto method = cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE;
    cv::findContours(binary, contours, mode, method);
    return contours;
}
cv::Mat rectang_detecter::highlight_blue_or_red(const cv::Mat &image, bool detect_blue)
{
    // 由于OpenCV的原因, 图像是BGR保存格式
    std::vector<cv::Mat> bgr_channel;
    cv::split(image, bgr_channel);

    //imshow("b",bgr_channel[0]);
    //imshow("g",bgr_channel[1]);
    //imshow("r",bgr_channel[2]);
    // 如果匹配蓝色
    if (detect_blue)
    {
        // 蓝通道减去绿通道
        return bgr_channel[0] - bgr_channel[1];
    }
    // 如果匹配红色
    else
    {
        // 红通道减去绿通道
        return bgr_channel[2] - bgr_channel[1];
    }
}
double rectang_detecter::adjust_threshold_binary(const cv::Mat &image,double threshold,double threshold_line)
{
	int nr= image.rows; // number of rows  hang
	int nc= image.cols * image.channels(); // total number of elements per line  lie
	float totalgray=0;
	const uchar* data;
	for (int j=0; j<nr; j++) {
	          data = image.ptr<uchar>(j);
	          for (int i=0; i<nc; i++) {
	        	  totalgray += (*data++)&1;
	            }
	      }
	threshold += (totalgray - threshold_line)/11000;
	threshold = threshold>255?255:threshold;
	threshold = threshold<0?0:threshold;
	return threshold;
}
std::vector<cv::RotatedRect> rectang_detecter::point_to_rects(const std::vector<std::vector<cv::Point>> &contours)
{
	std::vector<cv::RotatedRect> rects; // 代表灯柱的矩形
	for(unsigned int i = 0; i < contours.size(); ++i)
	{
		rects.push_back(cv::minAreaRect(contours[i]));
	}
	return rects;
}
std::vector<cv::RotatedRect> rectang_detecter::to_light_rects(const std::vector<std::vector<cv::Point>> &contours_light, const std::vector<std::vector<cv::Point>> &contours_brightness)
{
    speed_test_reset();

    // 创建矩形并预留空间
    std::vector<cv::RotatedRect> lights; // 代表灯柱的矩形
    lights.reserve(contours_brightness.size());

    // 遍历所有轮廓判断颜色轮廓(膨胀后的轮廓)是否在灯轮廓内
    std::vector<int> is_processes(contours_brightness.size()); // 保存灰度图轮廓是否已经确定是灯的轮廓
    for (unsigned int i = 0; i < contours_light.size(); ++i)
    {
        for (unsigned int j = 0; j < contours_brightness.size(); ++j)
        {
            // 如果当前轮廓没有确定是灯柱的轮廓
            if (!is_processes[j])
            {
                // 如果颜色(红色/蓝色)轮廓在灰度图轮廓内
                if (cv::pointPolygonTest(contours_brightness[j], contours_light[i][0], true) >= 0.0)
                {
                    // 转换成可旋转矩形并添加到容器中
                    lights.push_back(cv::minAreaRect(contours_brightness[j]));

                    // 设置当前轮廓已经确定是灯的轮廓
                    is_processes[j] = true;
                    break;
                }
            }
        }
    }

    speed_test_end("矩形拟合: ", "ms");

    return lights;
}
std::vector<cv::RotatedRect> rectang_detecter::detect_lights(bool detect_blue)
{

	auto light = highlight_blue_or_red(m_image, detect_blue); // 灯柱周边颜色高亮图

	static float thresh_binary=170;
	cv::threshold(m_gray, m_binary_brightness, thresh_binary, 255,
			cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
	thresh_binary = adjust_threshold_binary(m_binary_brightness,thresh_binary,threshold_line_binary);
	thresh_binary = thresh_binary>160?160:thresh_binary;
	thresh_binary = thresh_binary<110?110:thresh_binary;
	std::cout<<"thresh_binary:"<<thresh_binary<<std::endl;

	double thresh = detect_blue ? 50 : 80;
	static float thresh_binary_color=50;
	cv::threshold(light, m_binary_color, thresh_binary_color, 255,
			cv::ThresholdTypes::THRESH_BINARY); // 蓝色/红色二值图
	thresh_binary_color = adjust_threshold_binary(m_binary_color,thresh_binary_color,threshold_line_binary_color);
	thresh_binary_color=thresh_binary_color>90?90:thresh_binary_color;
	thresh_binary_color=thresh_binary_color<30?30:thresh_binary_color;
	imshowd("m_binary_light", m_binary_color);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(m_binary_color, m_binary_color, element, cv::Point(-1, -1), 1);
	m_binary_light = m_binary_color & m_binary_brightness; // 两二值图交集

#ifdef _DEBUG_VISION
	auto contours_light = find_contours(m_binary_light.clone()); // 两二值图交集后白色的轮廓
#else
	auto contours_light = find_contours(m_binary_light); // 两二值图交集后白色的轮廓
#endif
#ifdef _DEBUG_VISION
	auto contours_brightness = find_contours(m_binary_brightness.clone()); // 灰度图灯轮廓
#else
	auto contours_brightness = find_contours(m_binary_brightness); // 灰度图灯轮廓
#endif

	cv::Mat result = m_image_hql.clone();
	cv::Mat result0 = m_image_hql.clone();
	drawContours(m_binary_brightness, contours_brightness, -1, cv::Scalar(127), 1);
	drawContours(m_binary_light, contours_light, -1, cv::Scalar(127), 1);
	imshowd("m_binary_brightness", m_binary_brightness);
	imshowd("m_binary_color", m_binary_light);
	return to_light_rects(contours_light, contours_brightness);
}
std::vector<cv::RotatedRect> rectang_detecter::filter_lights(const std::vector<cv::RotatedRect> &lights, float thresh_max_angle, float thresh_min_area, float thresh_max_area)
{
    std::vector<cv::RotatedRect> rects;
    rects.reserve(lights.size());

	//std::cout<<"size: "<<lights.size()<<std::endl;
	for (const auto &rect : lights) {
		auto angle = 0.0f;
		auto whratio = rect.size.width / rect.size.height;

		if (whratio > 3.0f / 2.0f|| whratio < 2.0f / 3.0f) {
			float w, h;
			if (abs(rectlongLean(rect, w, h)) < thresh_max_angle) {

				//std::cout<<"Angle: "<<angle<<std::endl;
				if (rect.size.area() >= thresh_min_area
						&& rect.size.area() <= thresh_max_area) {
					rects.push_back(rect);
				}
			}
		}
	}

    return rects;
}
float rectang_detecter::point_distance(const cv::Point2f point,const cv::Point2f center){
	auto dx = point.x - center.x;
	auto dy = point.y - center.y;
	return std::sqrt(dx*dx+dy*dy);
}
float rectang_detecter::rectlongLean(const cv::RotatedRect &rect,float &w,float &h){
	cv::Point2f vertex[4];
	rect.points(vertex);
	cv::Point2f vertex_orderd[4];
	w = point_distance(vertex[0],vertex[1]);
	h = point_distance(vertex[1],vertex[2]);
	float angle = 0;
	if (w > h)
	{
		std::swap(w, h);
		//angle = std::abs(std::abs(std::atan2(vertex[1].y-vertex[0].y,vertex[1].x-vertex[0].x)) * 180/3.141592653-90);
		angle = atan2(-(vertex[1].y - vertex[0].y), vertex[1].x - vertex[0].x) * 180 / 3.141592653;
		angle = angle > 0 ? angle : angle + 180;
		angle = 90 - angle;
	}
	else
	{
		//angle = std::abs(std::abs(std::atan2(vertex[1].y-vertex[2].y,vertex[1].x-vertex[2].x)) * 180/3.141592653 - 90);
		angle = atan2(-(vertex[1].y - vertex[2].y), vertex[1].x - vertex[2].x) * 180 / 3.141592653;
		angle = angle > 0 ? angle : angle + 180;
		angle = 90 - angle;
	}
	return angle;
}
std::vector<rectandetect_info> rectang_detecter::detect_select_rect(const std::vector<cv::RotatedRect> &lights)
{
	std::vector<rectandetect_info> rectangs;
	//rectandetect_info rectang = rectandetect_info();
	for (const auto &light1 : lights) {
		for (const auto &light2 : lights) {

			float w1,h1,w2,h2,ang1,ang2,h_,w_,ang_;
			ang1 = rectlongLean(light1,w1,h1);
			ang2 = rectlongLean(light2,w2,h2);
			h_=(h1+h2)/2; w_=(w1+w2)/2; ang_=(ang1+ang2)/2;



			if(		abs(ang1-ang2)<20 &&
					abs(h1-h2) < 0.7*h_
					//&&abs(w1-w2)<w_/4
					)//delta h , w ,ang
			{

				if(abs(light1.center.x - light2.center.x) < 3.5*h_ && abs(light1.center.x - light2.center.x) > 0.8*h_
					&& abs(light1.center.y - light2.center.y) < 0.7*(1 + ang_/40)*h_ )    //delta x ,y
				{

		            cv::RotatedRect rect;
		            rect.angle = ang_;
		            rect.center.x = (light1.center.x + light2.center.x) / 2;
		            rect.center.y = (light1.center.y + light2.center.y) / 2;
		            rect.size.width = 1.2*abs(light1.center.x - light2.center.x);
		            rect.size.height = 1.5*h_;

		            float ww, hh ,angg;
		            angg = rectlongLean(rect, ww, hh);

					cv::Point2f vertex[4];
					rect.points(vertex);
					std::vector<cv::Point2f> vecpoint;

					for(auto (&vertexn):vertex)
					{
						vecpoint.push_back(vertexn);

					}

					float lost1 = abs(ang1-ang2)*h_/30+abs(h1-h2);
					float lost2 = abs(abs(light1.center.x - light2.center.x)-2.4*h_)+abs(light1.center.y - light2.center.y);
					float totallost = lost1 + lost2;
//					if (totallost < rectang.lost)
//					{
//						rectang = rectandetect_info(rect, vecpoint, light1,light2, totallost);
//					}
					rectangs.push_back(rectandetect_info(rect, vecpoint, light1,light2, totallost));
				}
				else
				{}

			}
			else
			{}

		}
	}


	return rectangs;
}
rectandetect_info* rectang_detecter::select_final_rectang(std::vector<rectandetect_info>& rectangs)
{
	std::sort(rectangs.begin(),rectangs.end(),[](const rectandetect_info &p1, const rectandetect_info &p2) { return p1.lost < p2.lost; });
	if(rectangs.size() != 0)
	{
		return &rectangs[0];
	}
	else
	{
		return old_finalrect;
	}
}

std::vector<cv::RotatedRect> rectang_detecter::detect_enemy_light(const cv::Mat &image, bool detect_blue)
{
    m_image = image.clone();
    m_image_hql = image.clone();
    m_show = image.clone();
    final_rectang = image.clone();
    possible_rectang = image.clone();
    light_img = image.clone();

    cv::cvtColor(m_image, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);//gray
    //imshowd("m_gray", m_gray);
    auto lights = detect_lights(detect_blue);

    draw_rotated_rects(light_img, lights, cv::Scalar(0,255,0), 2, true, cv::Scalar(255,0,0));

    lights = filter_lights(lights, m_threshold_max_angle, m_threshold_min_area,m_threshold_max_area);

    draw_rotated_rects(light_img, lights, cv::Scalar(0,0,255), 2, true, cv::Scalar(255,0,0));

    imshow("light_img",light_img);

    auto Rectang = detect_select_rect(lights);


	return lights;

}
bool rectang_detecter::detect(const cv::Mat &image, bool detect_blue)
{
    m_image = image.clone();
    m_image_hql = image.clone();
    m_show = image.clone();
    final_rectang = image.clone();
    possible_rectang = image.clone();
    light_img = image.clone();

    cv::cvtColor(m_image, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);//gray
    //imshowd("m_gray", m_gray);
    auto lights = detect_lights(detect_blue);

    draw_rotated_rects(light_img, lights, cv::Scalar(0,255,0), 2, true, cv::Scalar(255,0,0));

    lights = filter_lights(lights, m_threshold_max_angle, m_threshold_min_area,m_threshold_max_area);

    draw_rotated_rects(light_img, lights, cv::Scalar(0,0,255), 2, true, cv::Scalar(255,0,0));
    imshow("light_img",light_img);

    auto Rectangs = detect_select_rect(lights);

    finalrect=select_final_rectang(Rectangs);

    for(auto it : Rectangs)
    {
    		draw_rotated_rect(final_rectang, it.rect, cv::Scalar(250, 100, 0), 2);
    }
    draw_rotated_rect(final_rectang, finalrect->rect, cv::Scalar(0, 255, 47), 2);
    imshow("final_rectang",final_rectang);

	return false;

}
}
}
