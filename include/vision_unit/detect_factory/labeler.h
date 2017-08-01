#ifndef LABELER_H
#define LABELER_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "armor_detect.h"

namespace autocar {
namespace vision_mul {
    class labeler {
    public:
        labeler(std::string img_dir_, std::string label_img_dir_, std::string label_txt_dir_, int cols_=640, int rows_=480);
        ~labeler();
        void gen_sample(const cv::Mat &image, std::vector<armor_info> armors);

    private:
        std::string img_dir;
        std::string label_img_dir;
        std::string label_txt_dir;
        int rows;
        int cols;
        int count;
    };
}
}

#endif // LABELER_H
