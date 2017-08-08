#include "draw.h"
#define _DEBUG_VISION

#ifdef _DEBUG_VISION

void draw_rotated_rect(const cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness)
{
    cv::Point2f vertex[4];

    rect.points(vertex);
    for (int i=0; i<4; i++)
        cv::line(img, vertex[i], vertex[(i+1)%4], color, thickness);
}

void draw_rotated_rects(const cv::Mat &img, const std::vector<cv::RotatedRect> &rects, const cv::Scalar &color, int thickness, bool tab, const cv::Scalar &text_color)
{
    for (unsigned int i = 0; i < rects.size(); ++i)
    {
        draw_rotated_rect(img, rects[i], color, thickness);
        if (tab)
            cv::putText(img, std::to_string(i + 1), rects[i].center, CV_FONT_ITALIC, 0.5, text_color, 2, 8);
    }
}

void draw_circle(const cv::Mat &img, const cv::Point &center, int radius, const cv::Scalar &color, int thickness)
{
    cv::circle(img, center, radius, color, thickness);
}

void imshowd(const cv::String &winname, const cv::Mat &img)
{
    cv::imshow(winname, img);
}

#endif // _DEBUG_VISION
