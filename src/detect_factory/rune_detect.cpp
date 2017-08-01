#include "rune_detect.h"

RuneDetect::RuneDetect()
{
    num_code.insert(std::make_pair(1,  1));
    num_code.insert(std::make_pair(27, 2));
    num_code.insert(std::make_pair(30, 3));
    num_code.insert(std::make_pair(44, 4));
    num_code.insert(std::make_pair(54, 5));
    num_code.insert(std::make_pair(55, 6));
    num_code.insert(std::make_pair(28, 7));
    num_code.insert(std::make_pair(63, 8));
    num_code.insert(std::make_pair(62, 9));

    obj_p.push_back(cv::Point2f(0.0,   0.0));
    obj_p.push_back(cv::Point2f(300,   0.0));
    obj_p.push_back(cv::Point2f(300,   150));
    obj_p.push_back(cv::Point2f(0.0,   150));

    digital_obj_p.push_back(cv::Point2f(0.0,   0.0));
    digital_obj_p.push_back(cv::Point2f(50.0,  0.0));
    digital_obj_p.push_back(cv::Point2f(50.0,  110.0));
    digital_obj_p.push_back(cv::Point2f(0.0,   110.0));


    camera_matrix = cv::Mat(3, 3, CV_64F, cam);
    dist_coeffs   = cv::Mat(5, 1, CV_64F, dist_c);

    rvec = cv::Mat::ones(3,1,CV_64F);
    tvec = cv::Mat::ones(3,1,CV_64F);

    img_p.resize(4, cv::Point(0, 0));
    num_order.resize(4, 0);
    digital_img_p.resize(4, cv::Point(0, 0));
    //initialTmpList();
}

void RuneDetect::initialTmpList()
{
    cv::Mat tmp;
    cv::Mat gray_tmp;
    for(unsigned int i=1;i<10;i++)
    {
        std::string tmp_name;
        std::string index;
        std::stringstream ss;
        ss<<i;
        ss>>index;
        tmp_name = "/home/dji/rune_detect/data/" + index + ".png";
        std::cout<<tmp_name<<std::endl;

        tmp = cv::imread(tmp_name, -1);
        if(tmp.empty())
            std::cout<<"Can not open the template"<<std::endl;

        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        double max_area = 0.0;
        unsigned int max_area_index;
        cv::cvtColor(tmp, gray_tmp, CV_BGR2GRAY);
        cv::threshold(gray_tmp, gray_tmp, 100, 255, cv::THRESH_BINARY);
        //        std::cout<<gray_tmp.channels()<<std::endl;
        cv::findContours(gray_tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        for(unsigned int i = 0; i<contours.size();i++)
        {
            if(cv::contourArea(contours[i])>max_area)
            {
                max_area = cv::contourArea(contours[i]);
                max_area_index = i;
            }
        }
        cv::drawContours(tmp, contours, max_area_index, cv::Scalar(0, 0, 255));
        tmps_contour.push_back(contours[max_area_index]);
    }
    //    std::cout<<tmps_contour.size()<<std::endl;

}

void RuneDetect::templetContour(cv::Mat tmp)
{
    cv::Mat tmp_gray_img;
    cv::cvtColor(tmp, tmp_gray_img, CV_BGR2GRAY);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    double max_area = 0.0;
    unsigned int max_area_index;

    cv::threshold(tmp, tmp, 100, 255, cv::THRESH_BINARY);
    cv::findContours(tmp_gray_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    for(unsigned int i = 0; i<contours.size();i++)
    {
        if(cv::contourArea(contours[i])>max_area)
        {
            max_area = cv::contourArea(contours[i]);
            max_area_index = i;
        }
    }
    //cv::drawContours(tmp, contours, max_area_index, cv::Scalar(0,0,255));
    //cv::imshow("tmp", tmp);
    //cv::waitKey(0);
    tmp_contour = contours[max_area_index];
}

void RuneDetect::findContour(cv::Mat img)
{
    cv::Mat thresholded_img, thresholded_img_clone, gray_img;
    std::vector<cv::Mat> splited_img;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::RotatedRect> possible_rects;
    std::vector<cv::RotatedRect> rects;
    std::vector<cv::RotatedRect> num_rects;
    cv::cvtColor(img, gray_img, CV_BGR2GRAY);
    std::vector<unsigned int> possible_rects_index;
    std::vector<unsigned int> rect_index;

    //cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(7, 7));
    //cv::dilate(gray_img, gray_img, element);
    //cv::GaussianBlur(gray_img, gray_img, cv::Size(3, 3), 2);
    //    cv::dilate(gray_img, gray_img, element);
    //cv::split(img, splited_img);


    //cv::adaptiveThreshold(splited_img[2], thresholded_img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 5);
    cv::threshold(gray_img, thresholded_img, 100, 255, cv::THRESH_BINARY);
    thresholded_img_clone = thresholded_img.clone();
    //    cv::dilate(thresholded_img, thresholded_img, element);
    //    cv::dilate(thresholded_img, thresholded_img, element);
    //    cv::erode(thresholded_img, thresholded_img, element);
    //cv::Canny(gray_img, cannyed_img, 80, 200, 3);
    cv::findContours(thresholded_img_clone, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    //cv::imshow("1", cannyed_img);
    //    cv::imshow("thresholded_img", thresholded_img);
    //    cv::waitKey(10);
    rects.clear();
    for(unsigned int i = 0; i<contours.size();i++)
    {
        if(cv::contourArea(contours[i])>4000)
        {
            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
            rects.push_back(rect);
            double param = cv::matchShapes(tmp_contour, contours[i], CV_CONTOURS_MATCH_I1, 0);
            if(param < 1)
            {
                //cv::drawContours(img, contours, i, cv::Scalar(0,0,255), 3);
                possible_rects.push_back(rect);
                possible_rects_index.push_back(i);
            }
        }
    }
    std::sort(possible_rects.begin(), possible_rects.end(), [](cv::RotatedRect a, cv::RotatedRect b){return a.center.y >= b.center.y;});
    std::sort(rects.begin(), rects.end(), [](cv::RotatedRect a, cv::RotatedRect b){return a.center.y >= b.center.y;});
    for(unsigned int i = 0;possible_rects.size() > 9 && i < possible_rects.size() && num_rects.size() < 9;i++)
    {
        if(num_rects.size() == 0)
        {
            num_rects.push_back(possible_rects[i]);
            rect_index.push_back(possible_rects_index[i]);
            continue;
        }
        //for(unsigned int j = 0; j < num_rects.size(); j++)
        {
            //if(std::abs(possible_rects[i].center.x - num_rects[j].center.x) > 10)
            {
                num_rects.push_back(possible_rects[i]);
            }
        }
    }
    //    for(unsigned int i = 0;i < num_rects.size();i++)
    //    {
    //        cv::Point2f vertex[4];
    //        std::cout << num_rects.size()<<std::endl;
    //        num_rects[i].points(vertex);
    //        for (int i=0; i<4; i++)
    //            cv::line(img, vertex[i], vertex[(i+1)%4], cv::Scalar(255,0,0), 2);
    //    }
    std::cout << num_rects.size()<<std::endl;
    if(num_rects.size() == 9)
    {
        findDigitalNum(img, num_rects);
        img_vector.clear();
        for(unsigned int i = 0; i < num_rects.size(); i++)
        {
            //        std::vector<float> img_points;
            num_rects[i].points(&img_p[0]);
            setImagePoints(img_p);
            //        for(unsigned int j = 0; j<4; j++)
            //            img_points.push_back(img_p[]);
            cv::Mat transmtx = cv::getPerspectiveTransform(img_p, obj_p);
            //        cv::solvePnP(obj_p, img_p, camera_matrix, dist_coeffs, rvec, tvec);
            cv::Mat img_out;
            cv::warpPerspective(img, img_out, transmtx, cv::Size(280, 140), cv::INTER_NEAREST);
            //        cv::drawContours(thresholded_img, contours, rect_index[i], 255, -1);
            //        cv::Mat img_crop_i(28, 28, CV_8UC1, 255);
            //        cv::Mat img_crop_i = img(num_rects[i]);
            //cv::imshow("rot_m", img_out);
            img_vector.push_back(std::make_pair(img_out, num_rects[i].center));
        }
    }

}

void RuneDetect::setImagePoints(std::vector<cv::Point2f> vertices)
{
    cv::Point2f lu, ld, ru, rd;
    std::sort(vertices.begin(), vertices.end(), [](const cv::Point2f & p1, const cv::Point2f & p2) { return p1.x < p2.x; });
    if (vertices[0].y < vertices[1].y){
        lu = vertices[0];
        ld = vertices[1];
    }
    else{
        lu = vertices[1];
        ld = vertices[0];
    }
    if (vertices[2].y < vertices[3].y)	{
        ru = vertices[2];
        rd = vertices[3];
    }
    else {
        ru = vertices[3];
        rd = vertices[2];
    }
    img_p[0] = lu;
    img_p[1] = ru;
    img_p[2] = rd;
    img_p[3] = ld;
    //img_p.push_back()
    //img_p.clear();
    //      img_p.push_back(lu);
    //      img_p.push_back(ru);
    //      img_p.push_back(rd);
    //      img_p.push_back(ld);
    //    cv::Mat(armor_points).convertTo(img_points, CV_32F);
}

void RuneDetect::clearImgVector()
{
    img_vector.clear();
}

void RuneDetect::findDigitalNum(cv::Mat img, std::vector<cv::RotatedRect> num_rects)
{
    std::sort(num_rects.begin(), num_rects.end(), [](cv::RotatedRect a, cv::RotatedRect b){return a.center.x <= b.center.x;});
    double x1 = (num_rects.begin())->center.x;
    double x2 = (num_rects.end()-1)->center.x;
    std::sort(num_rects.begin(), num_rects.end(), [](cv::RotatedRect a, cv::RotatedRect b){return a.center.y <= b.center.y;});
    double y = (num_rects.begin())->center.y;
    double offset = (num_rects.begin()->size.height)/2;

    cv::Mat num_region = img(cv::Rect(cv::Point(x1 + offset, y - offset*4), cv::Point(x2 - offset, y - offset)));
    cv::Mat num_region_gray;
    cv::blur(num_region, num_region, cv::Size(3, 3));
    cv::cvtColor(num_region, num_region_gray, CV_BGR2GRAY);
    cv::threshold(num_region_gray, num_region_gray, 120, 255, cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(num_region_gray, num_region_gray, element);
    //cv::erode(num_region_gray, num_region_gray, element);


    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat num_contour_img = num_region_gray.clone();
    cv::findContours(num_contour_img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    img_rec = num_region_gray;
    numDecode(contours);

    //    cv::imshow("region", img_rec);
    //    cv::waitKey(150);
}

void RuneDetect::numDecode(std::vector<std::vector<cv::Point> > contours)
{
    std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> a, std::vector<cv::Point> b){return (a[0].x) <= (b[0]).x;});
    std::vector<std::vector<cv::Point>> ture_contours;
    for(unsigned int i = 0;i < contours.size(); i++)
    {
        double c1inc2 = 0;
        for(unsigned int j = 0;j < contours.size(); j++)
        {
            c1inc2 = cv::pointPolygonTest(contours[j], contours[i][0], false);
            if(c1inc2 > 0)
                break;
        }
        if(c1inc2 <=0)
            ture_contours.push_back(contours[i]);
    }
    //    double max_similar;
    //    unsigned int max_similar_index;
    //    for(unsigned int i = 0;i < ture_contours.size(); i++)
    //    {
    //        max_similar = INT_MAX;
    //        max_similar_index = 0;
    //        for(unsigned int j = 0;j < tmps_contour.size(); j++)
    //        {
    //            double param = cv::matchShapes(tmps_contour[j], ture_contours[i], CV_CONTOURS_MATCH_I1, 0);
    //            if(param < max_similar)
    //            {
    //                max_similar = param;
    //                max_similar_index = j + 1;
    //            }
    //        }
    //        num_order.push_back(max_similar_index);
    //        std::cout<<max_similar_index<<std::endl;
    //    }
    //    std::cout<<"********************************"<<std::endl;
    num_order.clear();
    std::cout<<std::endl<<"************************"<<std::endl;
    for(unsigned int i = 0;i < ture_contours.size();i++)
    {
        if(cv::contourArea(ture_contours[i]) > 50)
        {
            //cv::drawContours(img_rec, ture_contours, i, cv::Scalar(0, 0, 255));
            cv::Rect rec = cv::boundingRect(ture_contours[i]);
            //cv::rectangle(img_rec, rec, cv::Scalar(0, 0, 255));
            //cv::circle(img_rec, cv::Point(rec.x, rec.y), 15, cv::Scalar(0, 0, 255), -1);
            //            double t = rec.height/rec.width;
            //            std::cout<<"h/w: "<<t<<std::endl;
            if((double)rec.height/(double)rec.width >= 2.5)
            {
                num = 1;
                num_order.push_back(num_code[num]);
            }
            else
            {
                offset = rec.width/6;
                height_offset = rec.height/4;
                width_offset  = rec.width/2;
                x1 = rec.x + offset;
                y1 = rec.y + height_offset;
                x2 = rec.x + width_offset;
                y2 = rec.y + offset;
                x3 = rec.x + width_offset*2 - offset;
                y3 = rec.y + height_offset;
                x4 = rec.x + width_offset*2 - offset*2;
                y4 = rec.y + height_offset*3;
                x5 = rec.x + width_offset - offset*2;
                y5 = rec.y + height_offset*4 - offset/2;
                x6 = rec.x + offset;
                y6 = rec.y + height_offset*3;
                num = 0;
                a = 32;
                b = 0;
                region = img_rec.clone();
                cv::cvtColor(region, region, CV_GRAY2BGR);

                cv::circle(region, cv::Point(x1, y1), 1, cv::Scalar(255, 0, 0), 0);
                cv::circle(region, cv::Point(x2, y2), 1, cv::Scalar(255, 0, 255), 0);
                cv::circle(region, cv::Point(x3, y3), 1, cv::Scalar(0, 0, 255), 0);
                cv::circle(region, cv::Point(x4, y4), 1, cv::Scalar(0, 0, 255), 0);
                cv::circle(region, cv::Point(x5, y5), 1, cv::Scalar(0, 0, 255), 0);
                cv::circle(region, cv::Point(x6, y6), 1, cv::Scalar(0, 0, 255), 0);

                //std::printf("1: %d, %d: %u\n", x1, y1, img_rec.at<unsigned char>(y1, x1));
                //std::cout<<"a: "<<a<<std::endl;
                num = (int)img_rec.at<uchar>(y1, x1) > 0 ? num | a : num | b;
                a = a >> 1;
                //std::printf("2: %d, %d: %u\n", x2, y2, img_rec.at<unsigned char>(y2, x2));
                //std::cout<<"a: "<<(int)a<<std::endl;

                num = (int)img_rec.at<uchar>(y2, x2) > 0 ? num | a : num | b;
                a = a >> 1;
                //std::printf("3: %d, %d: %u\n", x3, y3, img_rec.at<unsigned char>(y3, x3));
                //std::cout<<"a: "<<a<<std::endl;
                num = (int)img_rec.at<uchar>(y3, x3) > 0 ? num | a : num | b;
                a = a >> 1;
                //std::printf("4: %d, %d: %u\n", x4, y4, img_rec.at<unsigned char>(y4, x4));
                //std::cout<<"a: "<<a<<std::endl;
                num = (int)img_rec.at<uchar>(y4, x4) > 0 ? num | a : num | b;
                a = a >> 1;
                //std::printf("5: %d, %d: %u\n", x5, y5, img_rec.at<unsigned char>(y5, x5));
                //std::cout<<"a: "<<a<<std::endl;
                num = (int)img_rec.at<uchar>(y5, x5) > 0 ? num | a : num | b;
                a = a >> 1;
                //std::printf("6: %d, %d: %u\n", x6, y6, img_rec.at<unsigned char>(y6, x6));
                //std::cout<<"a: "<<a<<std::endl;
                num = (int)img_rec.at<uchar>(y6, x6) > 0 ? num | a : num | b;
                //std::cout<<std::endl<<"num: "<<num<<std::endl;
                num_order.push_back(num_code[num]);
                region = region(rec);
                //std::cout<<std::endl<<"************************"<<std::endl;

                //            cv::RotatedRect rect = cv::minAreaRect(ture_contours[i]);
                //            cv::Point2f vertex[4];
                //            rect.points(vertex);
                //            for (int i=0; i<4; i++)
                //                cv::line(img_rec, vertex[i], vertex[(i+1)%4], cv::Scalar(255,0,0), 2);
                //            rect.points(&digital_img_p[0]);
                //            setImagePoints(digital_img_p);
                //        for(unsigned int j = 0; j<4; j++)
                //            img_points.push_back(img_p[]);
                //            cv::Mat transmtx = cv::getPerspectiveTransform(digital_img_p, digital_obj_p);
                //            cv::Mat img_out;
                //            cv::warpPerspective(img_rec, img_out, transmtx, cv::Size(50, 110), cv::INTER_NEAREST);
            }
            std::cout<<num_code[num];
            //cv::imshow("we", region);
            //cv::imshow("me", img_rec);
            //if(cv::waitKey(0)>0){}
        }
    }
    std::cout<<std::endl<<"************************"<<std::endl;

}

void RuneDetect::getNumOrder(std::vector<unsigned int> &list)
{
    list = num_order;
}

void RuneDetect::recogHandWriteNum()
{

}
