#include "armor_detect.h"
#include "debug_utility.hpp"
#include "draw.h"
#include <algorithm>
#include <cmath>
namespace autocar
{
namespace vision_mul
{
const float armor_detecter::m_threshold_max_angle = 15.0f;


const float armor_detecter::m_threshold_min_area = 5.0f;


const float armor_detecter::m_threshold_dis_lower = 1.0f;


const float armor_detecter::m_threshold_dis_upper = 3.0f;


const float armor_detecter::m_threshold_len = 3.0f;


const float armor_detecter::m_threshold_height = 2.0f;


const float armor_detecter::m_threshold_gray = 100.0f;

armor_info::armor_info()
{

}
armor_detecter::armor_detecter(bool debug_on)
{
    armor = new armor_info();
    debug_on_ = debug_on;
    old_armor = new armor_info();
    old_armor->armor.center.x = 0;
    old_armor->armor.size.width = 0;
    old_armor->armor.size.height = 0;
}


cv::Mat armor_detecter::highlight_blue_or_red(const cv::Mat &image, bool detect_blue)
{
    // 由于OpenCV的原因, 图像是BGR保存格式
    std::vector<cv::Mat> bgr_channel;
    cv::split(image, bgr_channel);
    cv::imshow("r",bgr_channel[2]);
    cv::imshow("g",bgr_channel[1]);
    cv::imshow("b",bgr_channel[0]);

    static cv::Mat r_his0,r_his1,r_his2;
    r_his2 = r_his1;
    r_his1 = r_his0;
    r_his0 = 0.7*(bgr_channel[2] - bgr_channel[1]*0.8) + 0.2 * r_his1 + 0.1*r_his2;
    cv::imshow("r-(g+b)",r_his0);

    // 如果匹配蓝色
    if (detect_blue)
    {
        // 蓝通道减去红通道
        return bgr_channel[0] - bgr_channel[1];
    }
    // 如果匹配红色
    else
    {
        // 红通道减去绿通道
        return r_his0;
    }


    //  cv::Mat img_hsv;
    //  cv::cvtColor(m_image, img_hsv, CV_BGR2HSV);

    //  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    //  //cv::morphologyEx(img_hsv, img_hsv , cv::MORPH_OPEN, element, cv::Point(-1,-1),3);
    //  //cv::morphologyEx(img_hsv, img_hsv , cv::MORPH_OPEN, element, cv::Point(-1,-1),3);

    //  if(detect_blue)
    //  {
    //    cv::Mat img_hsv_blue, img_threshold_blue;
    //    img_hsv_blue = img_hsv.clone();
    //    cv::Mat blue_low(cv::Scalar(60,43,46));
    //    cv::Mat blue_higher(cv::Scalar(140,255,255));
    //    cv::inRange(img_hsv_blue, blue_low, blue_higher, img_threshold_blue);
    //    //cv::imshow("img_threshold_blue", img_threshold_blue);
    //    return img_threshold_blue;
    //    //cv::waitKey(1);

    //    //auto contours_brightness = find_contours(img_threshold_blue);
    //  }
    //  else
    //  {
    //    cv::Mat img_hsv_red1, img_hsv_red2, img_threshold_red, img_threshold_red1, img_threshold_red2;
    //    img_hsv_red1  = img_hsv.clone();
    //    img_hsv_red2  = img_hsv.clone();
    //    cv::Mat red1_low(cv::Scalar(0,43,46));
    //    cv::Mat red1_higher(cv::Scalar(3,255,255));

    //    cv::Mat red2_low(cv::Scalar(170,43,46));
    //    cv::Mat red2_higher(cv::Scalar(180,255,255));
    //    cv::inRange(img_hsv_red1, red1_low, red1_higher, img_threshold_red1);
    //    cv::inRange(img_hsv_red2, red2_low, red2_higher, img_threshold_red2);
    //    img_threshold_red = img_threshold_red1 | img_threshold_red2;
    //    //cv::imshow("img_threshold_red", img_threshold_red);
    //    return img_threshold_red;

    //    //auto contours_brightness = find_contours(img_threshold_red);
    //    }
}

std::vector<std::vector<cv::Point>> armor_detecter::find_contours(const cv::Mat &binary)
{
    std::vector<std::vector<cv::Point>> contours; // 边缘
    const auto mode = cv::RetrievalModes::RETR_EXTERNAL;
    const auto method = cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE;
    cv::findContours(binary, contours, mode, method);
    return contours;
}

std::vector<cv::RotatedRect> armor_detecter::to_light_rects(const std::vector<std::vector<cv::Point>> &contours_light, const std::vector<std::vector<cv::Point>> &contours_brightness)
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
volatile int threshold1 = 180;
volatile int threshold2 = 50;
std::vector<cv::RotatedRect> armor_detecter::detect_lights(bool detect_blue)
{

    //  auto light = highlight_blue_or_red(m_image, detect_blue); // 灯柱周边颜色高亮图
    //  cv::threshold(m_gray, m_binary_brightness, 150, 255, cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
    //  //double thresh = detect_blue ? 120 : 50;
    //  //cv::threshold(light, m_binary_color, thresh, 255, cv::ThresholdTypes::THRESH_BINARY); // 蓝色/红色二值图
    //  //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //  //cv::dilate(m_binary_color, m_binary_color, element, cv::Point(-1, -1), 1);

    //  //m_binary_light = m_binary_color & m_binary_brightness; // 两二值图交集

    //#ifdef _DEBUG_VISION
    //  auto contours_light = find_contours(light.clone()); // 两二值图交集后白色的轮廓
    //#else
    //  auto contours_light = find_contours(m_binary_light); // 两二值图交集后白色的轮廓
    //#endif

    //#ifdef _DEBUG_VISION
    //  auto contours_brightness = find_contours(m_binary_brightness.clone()); // 灰度图灯轮廓
    //#else
    //  auto contours_brightness = find_contours(m_binary_brightness); // 灰度图灯轮廓
    //#endif

    //  return to_light_rects(contours_light, contours_brightness);
    // 亮图灯柱周边颜色
    //speed_test_reset();
    auto light = highlight_blue_or_red(m_image, detect_blue); // 灯柱周边颜色高亮图
    cv::threshold(m_gray, m_binary_brightness, threshold1, 255, cv::ThresholdTypes::THRESH_BINARY); // 亮度二值图
    double thresh = detect_blue ? 90 : 50;
    cv::threshold(light, m_binary_color, threshold2, 255, cv::ThresholdTypes::THRESH_BINARY); // 蓝色/红色二值图
//    cv::imshow("m_binary_brightness",m_binary_brightness);
//    cv::imshow("m_binary_color",m_binary_color);

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

    return to_light_rects(contours_light, contours_brightness);
}


std::vector<cv::RotatedRect> armor_detecter::filter_lights(const std::vector<cv::RotatedRect> &lights, float thresh_max_angle, float thresh_min_area)
{
    std::vector<cv::RotatedRect> rects;
    rects.reserve(lights.size());

    //std::cout<<"size: "<<lights.size()<<std::endl;
    for (const auto &rect : lights)
    {
        auto angle = 0.0f;
        auto whratio = rect.size.width / rect.size.height;
        if (whratio > 2.0f || whratio < 1.0f / 2.0f)
        {
            angle = rect.angle >= 45 ? std::abs(rect.angle - 45) : std::abs(rect.angle);
            angle = angle >= 45 ? std::abs(angle - 45) : std::abs(angle);
        }

        //std::cout<<"Angle: "<<angle<<std::endl;
        if (rect.size.area() >= thresh_min_area)
        {
            rects.push_back(rect);
        }
    }

    return rects;
}


std::vector<armor_info> armor_detecter::possible_armors(const std::vector<cv::RotatedRect> &lights, float thres_max_angle, float thres_dis_lower, float thres_dis_upper, float thres_len, float thres_height, float thres_gray, bool detect_blue)
{
    std::vector<armor_info> rects;
    rects.reserve(static_cast<std::vector<cv::RotatedRect>::size_type>(std::pow(lights.size(), 2)));

    for (const auto &light1 : lights)
    {
        for (const auto &light2 : lights)
        {
            auto edge1 = std::minmax(light1.size.width, light1.size.height);
            auto edge2 = std::minmax(light2.size.width, light2.size.height);
            auto distance = std::sqrt((light1.center.x-light2.center.x)*(light1.center.x-light2.center.x)+(light1.center.y-light2.center.y)*(light1.center.y-light2.center.y));
            auto center_angle = std::atan((light1.center.y - light2.center.y) / (light1.center.x - light2.center.x)) * 180 / CV_PI;
            //std::cout<<"center angle: "<<center_angle<<std::endl;

            cv::RotatedRect rect;
            rect.angle = static_cast<float>(center_angle);
            rect.center.x = (light1.center.x + light2.center.x) / 2;
            rect.center.y = (light1.center.y + light2.center.y) / 2;
            rect.size.width = static_cast<float>(distance)-std::max(edge1.first, edge2.first);
            rect.size.height = std::max(edge1.second, edge2.second);

            if (std::abs(center_angle) < thres_max_angle
                    && rect.size.area() > 100.0
                    && std::abs(light1.angle - light2.angle) < 10.0
                    && distance / std::max(edge1.second, edge2.second) < thres_dis_upper
                    && distance / std::max(edge1.second, edge2.second) > thres_dis_lower
                    && std::max(edge1.second, edge2.second) / std::min(edge2.second, edge1.second) <= thres_len
                    && std::abs(light1.center.y - light2.center.y) / std::max(edge1.second, edge2.second)  <= thres_height
                    && m_gray.at<uchar>(static_cast<int>(rect.center.y), static_cast<int>(rect.center.x)) < thres_gray)
            {
                int type = detect_blue ? 0 : 1;
                if (light1.center.x < light2.center.x)
                {
                    std::vector<cv::Point2f> armor_points;
                    cal_armor_info(armor_points, light1, light2);
                    rects.push_back(armor_info(type,rect, armor_points, light1, light2));
                    armor_points.clear();
                }
                else
                {
                    std::vector<cv::Point2f> armor_points;
                    cal_armor_info(armor_points, light2, light1);
                    rects.push_back(armor_info(type,rect, armor_points, light1, light2));
                    armor_points.clear();
                }
            }
        }
    }

    return rects;
}

void armor_detecter::cal_armor_info(std::vector<cv::Point2f> &armor_points, cv::RotatedRect left_light, cv::RotatedRect right_light)
{
    cv::Point2f left_points[4], right_points[4];
    left_light.points(left_points);
    right_light.points(right_points);
    cv::Point2f right_lu, right_ld, lift_ru, lift_rd;
    std::sort(left_points, left_points + 4, [](const cv::Point2f & p1, const cv::Point2f & p2) { return p1.x < p2.x; });
    std::sort(right_points, right_points + 4, [](const cv::Point2f & p1, const cv::Point2f & p2) { return p1.x < p2.x; });
    if (right_points[0].y < right_points[1].y){
        right_lu = right_points[0];
        right_ld = right_points[1];
    }
    else{
        right_lu = right_points[1];
        right_ld = right_points[0];
    }

    if (left_points[2].y < left_points[3].y)	{
        lift_ru = left_points[2];
        lift_rd = left_points[3];
    }
    else {
        lift_ru = left_points[3];
        lift_rd = left_points[2];
    }
    armor_points.push_back(lift_ru );
    armor_points.push_back(right_lu);
    armor_points.push_back(right_ld);
    armor_points.push_back(lift_rd );
}


std::vector<armor_info> armor_detecter::filter_by_features(std::vector<armor_info> &armors)
{
    cv::Mat mask = cv::Mat::zeros(m_gray.size(), CV_8UC1);

    std::vector<armor_info> new_armors;

    for (const auto &single_armor : armors)
    {
        auto &rect = single_armor.armor;
        cv::RotatedRect inner_box;
        inner_box.angle = rect.angle;
        inner_box.center = rect.center;
        inner_box.size.height = rect.size.height * 0.8f;
        if (rect.size.width >= 70)
        {
            inner_box.size.width = rect.size.width * 0.7f;
        }
        else if (rect.size.width <= 30.0f)
        {
            inner_box.size.width = rect.size.width * 0.3f;
        }
        else
        {
            inner_box.size.width = std::pow(rect.size.width, 2) * 0.01f;
        }
        draw_rotated_rect(m_show, inner_box, cv::Scalar(0,255,0), 2);

        CvPoint2D32f fpts[4];
        cv::Point pts[4];
        cvBoxPoints(CvBox2D(inner_box), fpts);
        for (int k=0; k<4; k++) {
            pts[k].x = fpts[k].x;
            pts[k].y = fpts[k].y;
        }
        cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255), 8, 0);

        cv::Mat mat_mean;
        cv::Mat mat_stddev;
        cv::meanStdDev(m_gray, mat_mean, mat_stddev, mask);

        auto stddev = mat_stddev.at<double>(0, 0);
        float limited_stddev = rect.size.area()>400 ? 40.0 : 4.0+(1600.0/((int)rect.size.area()%400));

        if (stddev < limited_stddev) {
            new_armors.push_back(single_armor);
            new_armors[new_armors.size()-1].score = stddev;
        }

        cv::fillConvexPoly(mask, pts, 4, cv::Scalar(0), 8, 0);
    }

    std::sort(new_armors.begin(), new_armors.end(), [](const armor_info &rect1, const armor_info &rect2)
    {
        return rect1.score < rect2.score;
    });

    // nms
    bool is_armor[new_armors.size()];
    for (int i=0; i<new_armors.size(); i++)
        is_armor[i] = true;
    for (int i=0; i<new_armors.size(); i++) {
        if (is_armor[i]) {
            for (int j=i+1; j<new_armors.size(); j++) {
                float dx = new_armors[i].armor.center.x-new_armors[j].armor.center.x;
                float dy = new_armors[i].armor.center.y-new_armors[j].armor.center.y;
                float dis = sqrt(dx*dx+dy*dy);
                if (dis < new_armors[i].armor.size.width+new_armors[j].armor.size.width)
                    is_armor[j] = false;
            }
        }
    }

    std::vector<armor_info> final_armors;
    for (int i=0; i<new_armors.size(); i++) {
        if (is_armor[i])
            final_armors.push_back(new_armors[i]);
    }

    return final_armors;
}
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
bool isCircle(const std::vector<cv::Point> &points,const cv::Point &center,float radius){
	if(radius > 7){
		const int perDegree = 15;
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
		bool r = true;
		for(int i = 0;i<flag.size();i++){
			r = r && flag[i];
		}
		kohill::print("finished.");
		return r;
	}
	return false;
}
bool armor_detecter::detect(const cv::Mat &image, bool detect_blue)
{
    //m_common = image.clone();

    std::vector<armor_info> all_armors;

    m_image = image.clone();
    if(debug_on_)
    {
        m_show = image.clone();
        possible_armor = image.clone();
        light_img = image.clone();
    }
    std::vector<cv::Vec3f> circles;
    cv::cvtColor(m_image, m_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    auto m_gray_circle = m_gray.clone();
    auto m_image_circle = m_image.clone();

    cv::Mat m_gray_binary ;
    cv::GaussianBlur(m_gray_circle, m_gray_circle, cv::Size(7, 7), 2, 2);
//    cv::threshold(m_gray_circle, m_gray_binary, 60, 255, cv::ThresholdTypes::THRESH_BINARY);
    cv::Canny(m_gray_circle,m_gray_binary,125,255);
    cv::imshow("th",m_gray_binary);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_gray_binary,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours.size() ; i++) {
		float radius = 0;
		cv::Point2f center;
		cv::minEnclosingCircle(cv::Mat(contours[i]), center, radius);
		if(isCircle(contours[i],center,radius)){
			cv::circle(m_image_circle, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
			cv::circle(m_image_circle, center, radius, cv::Scalar(255, 0, 0), 3,
					8, 0);
		}
	}

    auto lights = detect_lights(detect_blue);
    //std::cout<<"lights size"<<lights.size()<<std::endl;

    //draw_rotated_rects(m_binary_brightness, lights, cv::Scalar(100), 2, true, cv::Scalar(100));

    //speed_test_reset();

    lights = filter_lights(lights, m_threshold_max_angle, m_threshold_min_area);

    if(debug_on_)
        draw_rotated_rects(light_img, lights, cv::Scalar(0,0,255), 2, true, cv::Scalar(255,0,0));
    //  if (lights.size() <= 1)
    //  {
    //    return false;
    //  }

    auto armors = possible_armors(lights, m_threshold_max_angle, m_threshold_dis_lower, m_threshold_dis_upper, m_threshold_len, m_threshold_height, m_threshold_gray, detect_blue);
    std::cout<<"Possible armors size: "<<armors.size()<<std::endl;
    for(auto it: armors)
        draw_rotated_rect(possible_armor, it.armor, cv::Scalar(0,0,255), 2);
    //  if (armors.empty())
    //  {
    //    return false;
    //  }

    armors = filter_by_features(armors);
    //  if (armors.empty())
    //  {
    //    return false;
    //  }

    //  for (const auto &armor : armors)
    //  {
    //    draw_rotated_rect(m_image, armor.armor, cv::Scalar(0,0,255), 2);
    //  }
    // debug_on_ = true;
    std::cout<<"debug_on"<<debug_on_<<std::endl;
    if(debug_on_)
        debug_vision();

    cv::imshow("m_gray_binary",m_gray_binary);
    cv::imshow("m_image_circle",m_image_circle);

    if(!armors.empty())
    {
        slect_final_armor(armors);
        if(debug_on_)
        {
//            draw_rotated_rect(m_image, armor->armor, cv::Scalar(0,0,255), 2);
//            debug_vision();
        }
        return true;
    }
    else
        return false;
}

void armor_detecter::slect_final_armor(std::vector<armor_info> all_armors)
{
    std::sort(all_armors.begin(), all_armors.end(), [](const armor_info &p1, const armor_info &p2) { return p1.armor.size.area() > p2.armor.size.area(); });
    if(all_armors.size() > 1)
        //        *armor = abs(all_armors[0].armor.center.x - old_armor->armor.center.x) < abs(all_armors[1].armor.center.x - old_armor->armor.center.x)
        //            ? all_armors[0]: all_armors[1];
        *armor = abs(all_armors[0].armor.size.area() - old_armor->armor.size.area()) < abs(all_armors[1].armor.size.area() - old_armor->armor.size.area())
            ? all_armors[0]: all_armors[1];
    else
        *armor = all_armors[0];
    *old_armor = *armor;
}

void armor_detecter::debug_vision()
{
    imshowd("light_img", light_img);
    imshowd("possible_armor", possible_armor);
    imshowd("m_image", m_image);
    //imshowd("m_common", m_common);
    cv::waitKey(1);
}
armor_info* armor_detecter::get_armor()
{
    return armor;
}
}
}
