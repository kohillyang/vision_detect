#ifndef __DRAW_H__
#define __DRAW_H__

#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#define _DEBUG_VISION

#ifdef _DEBUG_VISION

void draw_rotated_rect(const cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness=1);
void draw_rotated_rects(const cv::Mat &img, const std::vector<cv::RotatedRect> &rects, const cv::Scalar &color, int thickness=1, bool tab=false, const cv::Scalar &text_color=cv::Scalar(100));
void draw_circle(const cv::Mat &img, const cv::Point &center, int radius, const cv::Scalar &color, int thickness=1);
void imshowd(const cv::String &winname, const cv::Mat &img);

#else

#define draw_rotated_rect(...)
#define draw_rotated_rects(...)
#define draw_circle(...)
#define imshowd(...)

#endif // _DEBUG_VISION


#endif // !__DRAW_H__
