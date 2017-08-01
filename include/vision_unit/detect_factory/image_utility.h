#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <opencv2/opencv.hpp>
#include <functional>
#include <algorithm>
#include <fstream>

#define if_less_equal(num, condition) if (num < condition) { num = condition; }
#define if_more_equal(num, condition) if (num > condition) { num = condition; }
#define if_more_swap(num1, num2) if (num1 > num2) { std::swap(num1, num2); }

namespace autocar
{
namespace vision_mul
{
	// 计算一条直线的角度, 范围 0-180 度
	// 参数: firstPt 第一个点
	// 参数: secondPt 第二个点
	// 返回值: 角度 (0-180)
	double calcLineDegree(const cv::Point2f& firstPt, const cv::Point2f& secondPt);

	// 将旋转矩形的角度转换为 0-180度
	// 参数: box 矩形
	// 返回值: 角度 (0-180)
	double getRcDegree(const cv::RotatedRect &box);

	// 计算直线的k、b
	// 参数: x1 横坐标1, x2 横坐标2
	// 参数: y1 纵坐标1, y2 纵坐标2
	// 返回值: 直线的参数k、b
	template <typename T>
	std::pair<double, double> get_linear_parameters(T x1, T y1, T x2, T y2);

	// 两点计算直线 k、b
	// 参数: point1 点1
	// 参数: point2 点2
	// 返回值: 直线的参数 k、b
	template <typename T>
	std::pair<double, double> get_linear_parameters(T point1, T point2);

	// 已知直线 k、b, 给定纵坐标 y 计算出坐标 x
	// 参数: coefficient 斜率 k
	// 参数: intercept 截距 b
	// 返回值: 横坐标 x 的值
	template <typename T>
	T get_linear_x(double coefficient, double intercept, T y);

	// 已知直线 k、b, 给定横坐标 x 计算出坐标 y
	// 参数: coefficient 斜率 k
	// 参数: intercept 截距 b
	// 返回值: 纵坐标 y 的值
	template <typename T>
	T get_linear_y(double coefficient, double intercept, T x);

	// 遍历旋转矩形内部的所有像素点
	// 参数: rect 旋转矩形
	// 参数: func 回调函数
	// 参数: xstep x坐标方向跳过点数
	// 参数: ystep y坐标方向跳过点数
	void for_each(const cv::RotatedRect &rect, const std::function<void(int, int)> &func, int xstep = 1, int ystep = 1);

	// 遍历旋转矩形内部的所有像素点
	// 参数: rect 旋转矩形
	// 参数: xmin x坐标最大值
	// 参数: ymin y坐标最大值
	// 参数: xmax x坐标最大值
	// 参数: ymax y坐标最大值
	// 参数: func 回调函数
	// 参数: xstep x坐标方向跳过点数
	// 参数: ystep y坐标方向跳过点数
	void for_each(const cv::RotatedRect &rect, float xmin, float ymin, float xmax, float ymax, const std::function<void(int, int)> &func, int xstep = 1, int ystep = 1);

	// 计算两点距离的平方
	// 参数: point1 第一个点
	// 参数: point2 第二个点
	// 返回值: 两点距离的平方
	template <typename T>
	double squaredist(const cv::Point_<T> &point1, const cv::Point_<T> &point2);

	// 计算两点距离的平方
	// 参数: x1 第一个点的x坐标
	// 参数: y1 第一个点的y坐标
	// 参数: x2 第二个点的x坐标
	// 参数: y2 第二个点的y坐标
	// 返回值: 两点距离的平方
	template <typename T>
	double squaredist(T x1, T y1, T x2, T y2);

	// 计算两点距离
	// 参数: point1 第一个点
	// 参数: point2 第二个点
	// 返回值: 两点距离
	template <typename T>
	double distance(const cv::Point_<T> &point1, const cv::Point_<T> &point2);

	// 计算两点距离
	// 参数: x1 第一个点的x坐标
	// 参数: y1 第一个点的y坐标
	// 参数: x2 第二个点的x坐标
	// 参数: y2 第二个点的y坐标
	// 返回值: 两点距离
	template <typename T>
	double distance(T x1, T y1, T x2, T y2);

	// 计算两点的中心点坐标
	// 参数: point1 第一个点
	// 参数: point2 第二个点
	// 返回值: 两点的中心点坐标
	template <typename T>
	T center_point(const T &point1, const T &point2);

	// 判断点与直线的距离
	// 参数: coefficient 直线斜率 k
	// 参数: intercept 直线截距 b
	// 参数: point 需要计算的点
	// 返回值: 点与直线的距离
	template <typename T>
	double point_to_line(double coefficient, double intercept, const T &point);

	// 判断点是否在另外两点构成的直线上
	// 参数: point1 直线点 1
	// 参数: point2 直线点 2 
	// 参数: point 将要判断的点
	// 返回值: true or false
	template <typename T>
	bool point_in_line(const T &point1, const T &point2, const T &point);
}
}

#endif // __UTILITY_H__
