#include <ros/ros.h>
#include <vision_unit/SetGoal.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "draw.h"
#include "debug_utility.hpp"
#include "armor_detect.h"
#include "armor_detect_node.h"
#include <geometry_msgs/PoseStamped.h>
#include <vision_unit/armor_msg.h>

int main(int argc, char **argv)
{

  ros::init(argc, argv, "armor_detect");
  try
  {
    //Try to detect armor.
    static autocar::vision_mul::armor_detect_node armor_solver;
    armor_solver.running();
  }
  catch (const std::exception &ex)
  {
    std::cout << ex.what() << std::endl;
  }
  return 0;
}







//#include "cv.h"
//#include "cxcore.h"
//#include "highgui.h"
//#include <iostream>
//
//using namespace std;
//int main()
//{
//    CvCapture* capture=cvCaptureFromCAM(-1);
//    CvVideoWriter* video=NULL;
//    IplImage* frame=NULL;
//    autocar::vision_mul::set_camera_exposure("/dev/video1", 100);
//    int n;
//    if(!capture) //如果不能打开摄像头给出警告
//    {
//       cout<<"Can not open the camera."<<endl;
//       return -1;
//    }
//    else
//    {
//       frame=cvQueryFrame(capture); //首先取得摄像头中的一帧
//        video=cvCreateVideoWriter("/home/kohill/kohillyang/camera1.avi", CV_FOURCC('X', 'V', 'I', 'D'), 25,
//       cvSize(frame->width,frame->height)); //创建CvVideoWriter对象并分配空间
// //保存的文件名为camera.avi，编码要在运行程序时选择，大小就是摄像头视频的大小，帧频率是32
//       if(video) //如果能创建CvVideoWriter对象则表明成功
//        {
//          cout<<"VideoWriter has created."<<endl;
//       }
//
//       cvNamedWindow("Camera Video",1); //新建一个窗口
//        int i = 0;
//       while(i <= 20000000) // 让它循环200次自动停止录取
//        {
//          frame=cvQueryFrame(capture); //从CvCapture中获得一帧
//           if(!frame)
//          {
//             cout<<"Can not get frame from the capture."<<endl;
//             break;
//          }
//          n=cvWriteFrame(video,frame); //判断是否写入成功，如果返回的是1，表示写入成功
//           cout<<n<<endl;
//          cvShowImage("Camera Video",frame); //显示视频内容的图片
//           i++;
//          if(cvWaitKey(2)>0)
//             break; //有其他键盘响应，则退出
//       }
//
//       cvReleaseVideoWriter(&video);
//       cvReleaseCapture(&capture);
//       cvDestroyWindow("Camera Video");
//    }
//    return 0;
// }
