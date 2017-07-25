#include "labeler.h"
#include "debug_utility.hpp"
#include <stdio.h>

namespace autocar
{
namespace vision_mul
{
labeler::labeler(std::string img_dir_, std::string label_img_dir_, std::string label_txt_dir_, int cols_, int rows_) {
    img_dir = img_dir_;
    label_img_dir = label_img_dir_;
    label_txt_dir = label_txt_dir_;
    rows =rows_;
    cols = cols_;
    count = 0;
}

labeler::~labeler() {
}

void labeler::gen_sample(const cv::Mat &image, std::vector<armor_info> armors) {
    count++;

    int r = image.rows;
    int c = image.cols;

    cv::Mat src_img;
    if (r!=rows || c!=cols)
        cv::resize(image, src_img, cv::Size(cols, rows));
    else
        src_img = image.clone();
    cv::Mat label_img = src_img.clone();

    char fpath[128];
    sprintf(fpath, "%s/%06d.txt", label_txt_dir.c_str(), count);
    FILE *fp = fopen(fpath, "w");
    for (const auto &armor : armors) {
        cv::Point2f pts[8];
        armor.left_light.points(pts);
        armor.right_light.points(pts+4);
        cv::Rect bb = cv::boundingRect(std::vector<cv::Point2f>(pts, pts+8));
        fprintf(fp, "%d %.6f %.6f %.6f %.6f\n",
                armor.type,
                (bb.x+bb.width/2.0f)/c,
                (bb.y+bb.height/2.0f)/r,
                (float)bb.width/c,
                (float)bb.height/r);

        cv::rectangle(label_img, cv::Rect((int)(bb.x*(float)cols/c), (int)(bb.y*(float)rows/r), (int)(bb.width*(float)cols/c), (int)(bb.height*(float)rows/r)), cv::Scalar(0,0,255), 2);
    }
    fclose(fp);
    console_log("  save to %s\n", fpath);

    sprintf(fpath, "%s/%06d.jpg", img_dir.c_str(), count);
    cv::imwrite(fpath, src_img);
    console_log("  save to %s\n", fpath);

    sprintf(fpath, "%s/%06d.jpg", label_img_dir.c_str(), count);
    cv::imwrite(fpath, label_img);
    console_log("  save to %s\n", fpath);
}
}
}
