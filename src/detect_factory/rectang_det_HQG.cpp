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
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
//#include "digital_classification.hpp"

#define _DEBUG_HQG
using namespace std;
#define _DEBUG_HQG_final

//Classifier myclassifier;

namespace autocar
{
namespace vision_mul
{
const float rectang_detecter::m_threshold_max_angle = 30.0f;

const float rectang_detecter::m_threshold_min_area = 2.0f;

const float rectang_detecter::m_threshold_max_area = 4000.0f;

const float rectang_detecter::threshold_line_binary = 5500.f;

const float rectang_detecter::threshold_line_binary_color = 4000.f;



rectangdetect_info::detectstate rectangdetect_info::state = rectangdetect_info::noneenermy;
cv::RotatedRect rectangdetect_info::car_rect = cv::RotatedRect();
int rectangdetect_info::car_light_num = 0;
float rectangdetect_info::rectdistance = 0.0;

rectangdetect_info::rectangdetect_info()
{
	//type = false;
	rect = cv::RotatedRect();
	left_light = cv::RotatedRect();
	right_light = cv::RotatedRect();
	score = 0.0f;
	lost = 10000.0;

}
rectang_detecter::rectang_detecter(bool debug_on)
{
	finalrect = new rectangdetect_info();
	debug_on_ = debug_on;
	old_finalrect = new rectangdetect_info();
	old_finalrect->rect.center.x = 0;
	old_finalrect->rect.size.width = 0;
	old_finalrect->rect.size.height = 0;
}
std::vector<std::vector<cv::Point>> rectang_detecter::find_contours(
		const cv::Mat &binary,int Mode)
{
	std::vector<std::vector<cv::Point>> contours; // 边缘
	const auto mode = cv::RetrievalModes::RETR_EXTERNAL; //RETR_LIST;
	const auto method = cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE;
	cv::findContours(binary, contours, mode, method);
	return contours;
}
cv::Mat rectang_detecter::highlight_blue_or_red(const cv::Mat &image,
		bool detect_blue)
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
double rectang_detecter::adjust_threshold_binary(const cv::Mat &image,
		double threshold, double threshold_line)
{
	int nr = image.rows; // number of rows  hang
	int nc = image.cols * image.channels(); // total number of elements per line  lie
	float totalgray = 0;
	const uchar* data;
	for (int j = 0; j < nr; j++)
	{
		data = image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++)
		{
			totalgray += (*data++) & 1;
		}
	}
	threshold += (totalgray - threshold_line) / 7000;
	threshold = threshold > 255 ? 255 : threshold;
	threshold = threshold < 0 ? 0 : threshold;
	return threshold;
}
std::vector<cv::RotatedRect> rectang_detecter::point_to_rects(
		const std::vector<std::vector<cv::Point>> &contours)
{
	std::vector<cv::RotatedRect> rects; // 代表灯柱的矩形
	for (unsigned int i = 0; i < contours.size(); ++i)
	{
		rects.push_back(cv::minAreaRect(contours[i]));
	}
	return rects;
}
std::vector<cv::RotatedRect> rectang_detecter::to_light_rects(
		const std::vector<std::vector<cv::Point>> &contours_light,
		const std::vector<std::vector<cv::Point>> &contours_brightness)
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
				if (cv::pointPolygonTest(contours_brightness[j],
						contours_light[i][0], true) >= 0.0)
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

	static float thresh_binary = 160;
	cv::threshold(m_gray, m_binary_brightness, thresh_binary, 255,
			cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
	thresh_binary = adjust_threshold_binary(m_binary_brightness, thresh_binary,
			threshold_line_binary);
	if(detect_blue)
	{thresh_binary = thresh_binary > 165 ? 165 : thresh_binary;thresh_binary = thresh_binary < 155 ? 155 : thresh_binary;}
	else
	{thresh_binary = thresh_binary > 155 ? 155 : thresh_binary;thresh_binary = thresh_binary < 150 ? 150 : thresh_binary;}

	double thresh = detect_blue ? 50 : 80;
	static float thresh_binary_color = 50;
	cv::threshold(light, m_binary_color, thresh_binary_color, 255,
			cv::ThresholdTypes::THRESH_BINARY); // 蓝色/红色二值图




	thresh_binary_color = adjust_threshold_binary(m_binary_color,
			thresh_binary_color, threshold_line_binary_color);
	if(detect_blue)
	{thresh_binary_color = thresh_binary_color > 50 ? 50 : thresh_binary_color;thresh_binary_color = thresh_binary_color < 30 ? 30 : thresh_binary_color;}
	else
	{thresh_binary_color = thresh_binary_color > 60 ? 60 : thresh_binary_color;thresh_binary_color = thresh_binary_color < 40 ? 40 : thresh_binary_color;}


	std::cout << "thresh_binary:" << thresh_binary << "thresh_binary_color:"
			<< thresh_binary_color << std::endl;

	//imshowd("m_binary_color0", m_binary_color);
	cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(m_binary_color, m_binary_color, element1, cv::Point(-1, -1), 1);
	m_binary_light = m_binary_color & m_binary_brightness; // 两二值图交集

	cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	cv::dilate(m_binary_brightness, m_binary_brightness, element2, cv::Point(-1, -1), 1);

#ifdef _DEBUG_HQG // _DEBUG_VISION
	auto contours_light = find_contours(m_binary_light.clone(),cv::RetrievalModes::RETR_EXTERNAL); // 两二值图交集后白色的轮廓
#else
			auto contours_light = find_contours(m_binary_light,cv::RetrievalModes::RETR_EXTERNAL); // 两二值图交集后白色的轮廓
#endif
#ifdef _DEBUG_HQG //_DEBUG_VISION
	auto contours_brightness = find_contours(m_binary_brightness.clone(),cv::RetrievalModes::RETR_EXTERNAL); // 灰度图灯轮廓
#else
			auto contours_brightness = find_contours(m_binary_brightness,cv::RetrievalModes::RETR_EXTERNAL); // 灰度图灯轮廓
#endif

#ifdef _DEBUG_HQG
	cv::Mat result = m_image_hql.clone();
	cv::Mat result0 = m_image_hql.clone();
	//drawContours(m_binary_brightness, contours_brightness, -1, cv::Scalar(127), 1);
	//drawContours(m_binary_light, contours_light, -1, cv::Scalar(127), 1);
	imshowd("m_gray", m_gray);
	imshowd("m_binary_color1", m_binary_color);
	imshowd("m_binary_brightness", m_binary_brightness);
	imshowd("m_binary_light", m_binary_light);
	cv::waitKey(1);
#endif
	return to_light_rects(contours_light, contours_brightness);
}
bool rectang_detecter::check_light_in_Mat(const cv::Mat &image,const cv::RotatedRect &light,float para1,float para2,bool detect_blue)
{
	cv::Point2f vertex[4];
	light.points(vertex);
	float bright = 0;
	int x,y,devide,count = 0,countbig = 0,total = 20;
	for(devide = total/8;devide <= total*7/8 ; devide++)
	{
		x = vertex[0].x*((float)devide/total) + vertex[2].x*(1.0-(float)devide/total);
		y = vertex[0].y*((float)devide/total) + vertex[2].y*(1.0-(float)devide/total);
		if(detect_blue)
		{
			bright += image.at<cv::Vec3b>(y,x)[0];
			if(image.at<cv::Vec3b>(y,x)[0]>190)
				countbig++;
		}
		else
		{
			bright += image.at<cv::Vec3b>(y,x)[2];
			if(image.at<cv::Vec3b>(y,x)[2]>190)
				countbig++;
		}
		count++;
	}
	if(bright/count>para1 || countbig>=total*para2)
		return true;
	else
		return false;
}
std::vector<cv::RotatedRect> rectang_detecter::filter_lights(
		const std::vector<cv::RotatedRect> &lights, bool detect_blue,float thresh_max_angle,
		float thresh_min_area, float thresh_max_area)
{
	std::vector<cv::RotatedRect> rects;
	rects.reserve(lights.size());

	float thread;
	float w, h, angle;

	//std::cout<<"size: "<<lights.size()<<std::endl;
	for (const auto &rect : lights)
	{
		angle = rectlongLean(rect, w, h);
		auto hwratio = h / w;

		if (h >= 3)
		{
			if (hwratio > 1.4 - 1.304*0.4 / sqrt(sqrt(h)))
			{
				if (abs(angle) < (thresh_max_angle + 15.0))
				{
					if (rect.size.area() >= thresh_min_area
							&& rect.size.area() <= thresh_max_area)
					{
						if(check_light_in_Mat(m_image,rect,145,0.1,detect_blue))
						{
							rects.push_back(rect);
							continue;
						}
					}
				}
			}
		}
		if(h<15)
		{
			if(hwratio<1.5)
			{
				if(check_light_in_Mat(m_image,rect,160,0.3,detect_blue))
				{
					cv::RotatedRect newrect;
					newrect.center = rect.center;
					newrect.size.width = rect.size.height/2;
					newrect.size.height = rect.size.height;
					newrect.angle = 0;
					rects.push_back(newrect);
					continue;
				}
			}
		}
	}

	return rects;
}
float rectang_detecter::point_distance(const cv::Point2f point,
		const cv::Point2f center)
{
	auto dx = point.x - center.x;
	auto dy = point.y - center.y;
	return std::sqrt(dx * dx + dy * dy);
}
float rectang_detecter::rectlongLean(const cv::RotatedRect &rect, float &w,
		float &h)
{
	cv::Point2f vertex[4];
	rect.points(vertex);
	cv::Point2f vertex_orderd[4];
	w = point_distance(vertex[0], vertex[1]);
	h = point_distance(vertex[1], vertex[2]);
	float angle = 0;
	if (w > h)
	{
		std::swap(w, h);
		//angle = std::abs(std::abs(std::atan2(vertex[1].y-vertex[0].y,vertex[1].x-vertex[0].x)) * 180/3.141592653-90);
		angle = atan2(-(vertex[1].y - vertex[0].y), vertex[1].x - vertex[0].x)
				* 180 / 3.141592653;
		angle = angle > 0 ? angle : angle + 180;
		angle = 90 - angle;
	}
	else
	{
		//angle = std::abs(std::abs(std::atan2(vertex[1].y-vertex[2].y,vertex[1].x-vertex[2].x)) * 180/3.141592653 - 90);
		angle = atan2(-(vertex[1].y - vertex[2].y), vertex[1].x - vertex[2].x)
				* 180 / 3.141592653;
		angle = angle > 0 ? angle : angle + 180;
		angle = 90 - angle;
	}
	return angle;
}
bool rectang_detecter::detect_two_light(const cv::RotatedRect &light1,
		const cv::RotatedRect &light2)
{
	float w1, h1, w2, h2, ang1, ang2, h_, w_, ang_;
	ang1 = rectlongLean(light1, w1, h1);
	ang2 = rectlongLean(light2, w2, h2);
	h_ = (h1 + h2) / 2;
	w_ = (w1 + w2) / 2;
	ang_ = (ang1 + ang2) / 2;

	if (fabs(h1 - h2) < h_)
	{
		if ((fabs(light1.center.x - light2.center.x) < 4.5 * h_
				&& fabs(light1.center.x - light2.center.x) + fabs(light1.center.y - light2.center.y) > 0.3 * h_
				&& fabs(light1.center.y - light2.center.y) < 2.0 * (1 + fabs(ang_ )/ 40) * h_) //delta x ,y)
			||(h_<15
			&&fabs(light1.center.x - light2.center.x) < 6.0 * h_
			&& fabs(light1.center.x - light2.center.x) + fabs(light1.center.y - light2.center.y) > 0.3 * h_
			&& fabs(light1.center.y - light2.center.y) < 2.5 * (1 + fabs(ang_ )/ 40) * h_) )
		{
			return true;
		}
	}

	return false;
}
cv::RotatedRect rectang_detecter::get_minrect_from_rects(std::vector<cv::RotatedRect> rects, float &error)
{
	cv::RotatedRect rect = cv::RotatedRect();
	cv::Point2f vertex[4];
	std::vector<cv::Point2f> points;
	if (rects.size() != 0)
	{
		for (auto it : rects)
		{
			it.points(vertex);
			for (int i = 0; i < 4; i++)
			{
				points.push_back(*(vertex + i));
			}
		}
		rect = cv::minAreaRect(points);

		/*计算error*/
		float sum = 0;
		float accum = 0;
		for (auto it : rects)
			sum += std::max(it.size.height,it.size.width);

		float mean = sum / (float)rects.size();   //均值

		//for (auto it : rects)
		//	accum += (std::max(it.size.height,it.size.width) - mean) * (std::max(it.size.height,it.size.width) - mean);

		//error = sqrt(accum / (float) rects.size());  //标准差
		float err;
		error = 0.5;

		for (auto it : rects)
		{
			err = fabs((float)(std::max(it.size.height,it.size.width))/mean - 1) - 0.3;
			error += err > 0 ? err : 0;
		}

		switch (rects.size())
		{
		case 3:
			error = 2 * error;
			break;
		case 4:
			break;
		default:
			std::cout << "rect.size() is error as" << (float) rects.size()
					<< std::endl;
			break;
		}
	}
	return rect;
}
float rectang_detecter::getlocalaveragegray(const cv::Mat& image ,float devidex,float devidey)
{
	int nr = image.rows; // number of rows  hang
	int nc = image.cols * image.channels(); // total number of elements per line  lie
	float totalgray = 0;
	int count=0;
	const uchar* data;
	for (int j = (int)(nr*0.5 - nr*devidey*0.5); j < (int)(nr*0.5 + nr*devidey*0.5); j++)
	{
		data = image.ptr < uchar > (j);
		for (int i = (int)(nc*0.5 - nc*devidex*0.5); i < (int)(nc*0.5 + nc*devidex*0.5); i++)
		{
			totalgray += (*data++);
			count++;
		}
	}
	totalgray = totalgray/(count+1);
	return totalgray;
}
std::vector<rectangdetect_info> rectang_detecter::detect_select_rect(
		const std::vector<cv::RotatedRect> &lights)
{
	std::vector<rectangdetect_info> rectangs;

	rectangdetect_info::state = rectangdetect_info::noneenermy;
	rectangdetect_info::car_rect = cv::RotatedRect();
	rectangdetect_info::car_light_num = 0;
	//rectangdetect_info rectang = rectangdetect_info();

	float car_lights_lost1 = 100000;
	float car_lights_lost2 = 0;

	bool state1, state2;
	/*以下为检测车辆*/
	for (const auto &light1 : lights)
	{
		for (const auto &light2 : lights)
		{

			float w1, h1, w2, h2, ang1, ang2, h_, w_, ang_;
			ang1 = rectlongLean(light1, w1, h1);
			ang2 = rectlongLean(light2, w2, h2);
			h_ = (h1 + h2) / 2;
			w_ = (w1 + w2) / 2;
			ang_ = (ang1 + ang2) / 2;


			state1 = false, state2 = false;
			cv::RotatedRect light3_1;
			cv::RotatedRect light3_2;
			if (detect_two_light(light1, light2))
			{
				for (const auto &light3 : lights)
				{
					if (light3.center != light2.center
							&& detect_two_light(light3, light1))
					{
						light3_1 = light3;
						state1 = true;
						continue;
					}
					if (light3.center != light1.center
							&& detect_two_light(light3, light2))
					{
						light3_2 = light3;
						state2 = true;
						continue;
					}
				}
			}
			std::vector<cv::RotatedRect> LightRects;
			LightRects.push_back(light1);
			LightRects.push_back(light2);

			if (state1 == true || state2 == true)
			{
				if (state1 == true)
					LightRects.push_back(light3_1);
				if (state2 == true)
					LightRects.push_back(light3_2);

				cv::RotatedRect carrect = get_minrect_from_rects(LightRects,
						car_lights_lost2);

				if (car_lights_lost2 < car_lights_lost1)
				{
					car_lights_lost1 = car_lights_lost2;
					rectangdetect_info::car_rect = carrect;
					rectangdetect_info::car_light_num = 2+(int)state1+(int)state2;
				}
			}
			LightRects.clear();
		}
	}
	/*以下为检测装甲板*/

	for (const auto &light1 : lights)
	{
		for (const auto &light2 : lights)
		{

			float w1, h1, w2, h2, ang1, ang2, h_, w_, ang_;
			ang1 = rectlongLean(light1, w1, h1);
			ang2 = rectlongLean(light2, w2, h2);
			h_ = (h1 + h2) / 2;
			w_ = (w1 + w2) / 2;
			ang_ = (ang1 + ang2) / 2;

			/*以下为检测装甲板*/
			if ((fabs(ang1 - ang2) < 25 && fabs(h1 - h2) < 0.6 * h_
			//&&fabs(w1-w2)<w_/4
					)
				||(h_< 12
				&&fabs(ang1 - ang2) < 40 && fabs(h1 - h2) < 0.7 * h_
					))//delta h , w ,ang
			{

				if ((fabs(light1.center.x - light2.center.x) < 3.5 * h_
						&& fabs(light1.center.x - light2.center.x) > 1.3 * h_
						&& fabs(light1.center.y - light2.center.y)
								< 0.7 * (1 + fabs(ang_) / 40) * h_) //delta x ,y
						||(h_< 12
						&&fabs(light1.center.x - light2.center.x) < 4.5 * h_
						&& fabs(light1.center.x - light2.center.x) > 1.3 * h_
						&& fabs(light1.center.y - light2.center.y)< 1.0 * (1 + fabs(ang_) / 40) * h_
						))
				{

					cv::RotatedRect rect;
					rect.angle = ang_;
					rect.center.x = (light1.center.x + light2.center.x) / 2;
					rect.center.y = (light1.center.y + light2.center.y) / 2;
					rect.size.width = 1.0
							* abs(light1.center.x - light2.center.x);
					rect.size.height = 1.0 * h_;

					float ww, hh, angg;
					angg = rectlongLean(rect, ww, hh);

					cv::Point2f vertex[4];
					rect.points(vertex);
					std::vector<cv::Point2f> vecpoint;

					for (auto (&vertexn) : vertex)
					{
						vecpoint.push_back(vertexn);

					}

					float lost1 = fabs(ang1 - ang2) / 10 + fabs(h1 - h2)/h_ + fabs(ang_)/30;
					float lost2 = fabs(fabs(light1.center.x - light2.center.x)/h_ - 2.7)
							+ 1.5 * fabs(light1.center.y - light2.center.y)/h_;
					float lost3 = 0;
					int lx = rect.center.x - rect.size.width/2;int ly = rect.center.y - rect.size.height/2;
					int lw = rect.size.width; int lh = rect.size.height;
					lx = lx + 0.3*lw; ly = ly + 0.0*lh; lw = 0.4*lw; lh = 1.0*lh;
					if(0 <= lx && 0 <= ly && lw > 0 && lh > 0 && lx + lw <= m_image.cols && ly + lh <= m_image.rows)
					{
						cv::Rect a = cv::Rect(lx, ly, lw, lh);
						cv::Mat lrect = m_image(a);
						cv::cvtColor(lrect, lrect,cv::ColorConversionCodes::COLOR_BGR2GRAY);
						imshowd("lrect", lrect);
						lost3 = - getlocalaveragegray(lrect, 1, 1);
					}
					float totallost = lost1 + lost2 + lost3/10 - 0.5 * h_;

					rectangs.push_back(rectangdetect_info(rect, vecpoint, light1, light2,totallost));
				}
				else
				{
				}

			}
			else
			{
			}

		}
	}

	return rectangs;
}
rectangdetect_info* rectang_detecter::select_final_rectang(
		std::vector<rectangdetect_info>& rectangs)
{
	std::sort(rectangs.begin(), rectangs.end(),
			[](const rectangdetect_info &p1, const rectangdetect_info &p2)
			{	return p1.lost < p2.lost;});
	if (rectangs.size() != 0)
	{
		for(auto it1 : rectangs)
		{
			bool sl = false, sr = false;
			for (auto it2 : rectangs)
			{
				float deltay = it1.rect.center.y - it2.rect.center.y;
				float h_ = (it1.rect.size.height + it2.rect.size.height) / 2;
				if (fabs(deltay) < h_ * 1)
				{
					float deltax = it1.rect.center.x - it2.rect.center.x;
					if (deltax > 0.5 * h_ && deltax < 4 * h_)
						sl = true;
					if (deltax > -0.5 * h_ && deltax < -4 * h_)
						sr = true;
				}
				if (sl == true && sr == true)
				{
					swap(it1, rectangs[0]);
					break;
				}
			}
		}
		if(rectangdetect_info::car_light_num == 4)
		{

		}

		return &rectangs[0];
	}
	else
	{
		return old_finalrect;
	}
}
void rectang_detecter::calculate_distance(std::vector<rectangdetect_info> & rect_infos)
{
	float distance0,distance1,distance2;
	int size = (int)(rect_infos.size());
	if(size!=0)
	{
		distance0 = 2000/(rect_infos[0].left_light.size.height+rect_infos[0].right_light.size.height);
		distance1 = 3*2000/fabs(rect_infos[0].left_light.center.x - rect_infos[0].right_light.center.x);

		rectangdetect_info::rectdistance = rectangdetect_info::rectdistance * 0.96 + (distance0+distance1)/2 * 0.04;
	}
	else
	{}

}
void rectang_detecter::tracking(cv::Mat frame, cv::Mat &model, cv::Rect &trackBox)
{
    cv::Mat gray;
    cvtColor(frame, gray, CV_RGB2GRAY);

    cv::Rect searchWindow;
    searchWindow.width = trackBox.width * 3;
    searchWindow.height = trackBox.height * 3;
    searchWindow.x = trackBox.x + trackBox.width * 0.5 - searchWindow.width * 0.5;
    searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
    searchWindow &= cv::Rect(0, 0, frame.cols, frame.rows);

    cv::Mat similarity;
    matchTemplate(gray(searchWindow), model, similarity, CV_TM_CCOEFF_NORMED);

    double mag_r;
    cv::Point point;
    cv::minMaxLoc(similarity, 0, &mag_r, 0, &point);
    trackBox.x = point.x + searchWindow.x;
    trackBox.y = point.y + searchWindow.y;
    model = gray(trackBox);
}
std::vector<rectangdetect_info> rectang_detecter::detect_enemy( const cv::Mat &image, bool detect_blue)
{

	m_image = image;

#ifdef _DEBUG_HQG

	light_img = image.clone();
	m_image_hql = image.clone();
	m_show = image.clone();
#endif
#ifdef	_DEBUG_HQG_final
	//final_rectang = image.clone();
#endif


	cv::cvtColor(m_image, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY); //gray
	//imshowd("m_gray", m_gray);
	auto lights = detect_lights(detect_blue);

#ifdef _DEBUG_HQG
	draw_rotated_rects(light_img, lights, cv::Scalar(0, 255, 0), 1, false,
			cv::Scalar(255, 0, 0));
#endif

	lights = filter_lights(lights,detect_blue, m_threshold_max_angle, m_threshold_min_area, m_threshold_max_area);

	auto Rectangs = detect_select_rect(lights);

	finalrect = select_final_rectang(Rectangs);

	calculate_distance(Rectangs);

#ifdef _DEBUG_HQG
	draw_rotated_rects(light_img, lights, cv::Scalar(0, 0, 255), 1, false, cv::Scalar(255, 0, 0));
	imshow("light_img", light_img);
#endif

#ifdef	_DEBUG_HQG_final
	if(Rectangs.size()!=0)
	{
//		int s = (finalrect->rect.size.height + finalrect->rect.size.width) / 7 + 1;
//		int centerx = finalrect->rect.center.x;
//		int centery = finalrect->rect.center.y;
//		cv::Rect a = cv::Rect(centerx - s / 2, centery - s / 2, s, s);
//		cv::Mat arm = final_rectang(a);
//		imshow("arm", arm);
//		string str = myclassifier.classify_mnist(arm);
//		std::cout<<"                                                 the number is: "+str<<std::endl;
//		cv::putText(final_rectang, str, finalrect->rect.center, CV_FONT_ITALIC, 1.5, cv::Scalar(55, 250, 0), 5, 8);

	for (auto it : Rectangs)
	{
		draw_rotated_rect(m_image, it.rect, cv::Scalar(194, 158, 241), 1);
	}
	draw_rotated_rect(m_image, finalrect->rect, cv::Scalar(0, 255, 47), 1);
	//draw_rotated_rect(m_image, rectangdetect_info::car_rect, cv::Scalar(250,100,0), 2);
	string str = static_cast<ostringstream*>( &(ostringstream() << rectangdetect_info::rectdistance) )->str();
	//cv::putText(m_image, str , finalrect->rect.center, CV_FONT_ITALIC, 0.5, cv::Scalar(55, 250, 0), 5, 8);
	}
	std ::cout <<m_image.cols << m_image.rows <<std::endl;
	cv::Size dsize = cv::Size(640, 480);
	cv::Mat mg_resize;
	cv::resize(m_image, mg_resize, dsize);

	imshow("final_rectang", mg_resize);

#endif

	if(Rectangs.size()==0)
	{
	}


	return Rectangs;

}
bool rectang_detecter::detect(const cv::Mat &image, bool detect_blue)
{
	m_image = image.clone();
	m_image_hql = image.clone();
	m_show = image.clone();
	final_rectang = image.clone();
	light_img = image.clone();

	cv::cvtColor(m_image, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY); //gray
	//imshowd("m_gray", m_gray);

	auto lights = detect_lights(detect_blue);

	draw_rotated_rects(light_img, lights, cv::Scalar(0, 255, 100), 1, true,
			cv::Scalar(255, 0, 0));

	lights = filter_lights(lights,detect_blue, m_threshold_max_angle, m_threshold_min_area,
			m_threshold_max_area);

	draw_rotated_rects(light_img, lights, cv::Scalar(0, 100, 255), 1, false,
			cv::Scalar(255, 0, 0));
	imshow("light_img", light_img);

	auto Rectangs = detect_select_rect(lights);

	finalrect = select_final_rectang(Rectangs);

	for (auto it : Rectangs)
	{
		draw_rotated_rect(final_rectang, it.rect, cv::Scalar(250, 100, 0), 2);
	}
	draw_rotated_rect(final_rectang, finalrect->rect, cv::Scalar(0, 255, 47),2);



	imshow("final_rectang", final_rectang);

	return false;

}
}
}
