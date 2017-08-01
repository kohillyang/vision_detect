#ifndef RUNEDETECT_H
#define RUNEDETECT_H

#include <opencv2/opencv.hpp>
#include <unordered_map>

class RuneDetect
{
public:
    RuneDetect();
    void initialTmpList();
    void templetContour(cv::Mat tmp);
    void findContour(cv::Mat img);
    void recogHandWriteNum();
    void setImagePoints(std::vector<cv::Point2f> armor_points);
    void clearImgVector();
    void findDigitalNum(cv::Mat img, std::vector<cv::RotatedRect> num_rects);
    void numDecode(std::vector<std::vector<cv::Point>> contours);
    void getNumOrder(std::vector<unsigned int> &list);
    std::vector<std::pair<cv::Mat, cv::Point2f>> img_vector;
private:
    std::vector<std::vector<cv::Point>>  tmps_contour;
    std::unordered_map<unsigned int, unsigned int> num_code;
    std::vector<unsigned int> num_order;
    cv::Mat img_rec;
    std::vector<cv::Point> tmp_contour;
    std::vector<cv::Point2f> obj_p;
    //    cv::Point2f img_p[];
    std::vector<cv::Point2f> img_p;
    std::vector<cv::Point2f> digital_img_p;
    std::vector<cv::Point2f> digital_obj_p;

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat region;

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    double cam[9] = {839.923052, 0.0,        340.780730,
                     0.0,        837.671081, 261.766523,
                     0.0,        0.0,        1.0       };

    double dist_c[5] = {0.082613, 0.043275, 0.002486, -0.000823, 0.0};

    int offset, height_offset, width_offset;
    int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, num, a, b;
};

#endif // RUNEDETECT_H
