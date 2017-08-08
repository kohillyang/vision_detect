#include "image_utility.h"
#include <limits>

namespace autocar
{
namespace vision_mul
{
// 计算一条直线的角度, 范围 0-180 度
// 参数: firstPt 第一个点
// 参数: secondPt 第二个点
// 返回值: 角度 (0-180)
double calcLineDegree(const cv::Point2f& firstPt, const cv::Point2f& secondPt)
{
	double curLineAngle = 0.0f;
	if (secondPt.x - firstPt.x != 0)
	{
		curLineAngle = atan(static_cast<double>(firstPt.y - secondPt.y) / static_cast<double>(secondPt.x - firstPt.x));
		if (curLineAngle < 0)
		{
			curLineAngle += CV_PI;
		}
	}
	else
	{
		curLineAngle = CV_PI / 2.0f; //90度
	}
	return curLineAngle*180.0f / CV_PI;
}


// 将旋转矩形的角度转换为 0-180度
// 参数: box 矩形
// 返回值: 角度 (0-180)
double getRcDegree(const cv::RotatedRect &box)
{
	using namespace cv;
	double degree = 0.0f;
	Point2f vertVect[4];
	box.points(vertVect);
	//line 1
	const double firstLineLen = (vertVect[1].x - vertVect[0].x)*(vertVect[1].x - vertVect[0].x) +
		(vertVect[1].y - vertVect[0].y)*(vertVect[1].y - vertVect[0].y);
	//line 2
	const double secondLineLen = (vertVect[2].x - vertVect[1].x)*(vertVect[2].x - vertVect[1].x) +
		(vertVect[2].y - vertVect[1].y)*(vertVect[2].y - vertVect[1].y);
	if (firstLineLen > secondLineLen)
	{
		degree = calcLineDegree(vertVect[0], vertVect[1]);
	}
	else
	{
		degree = calcLineDegree(vertVect[2], vertVect[1]);
	}
	return degree;
}


// 遍历旋转矩形内部的所有像素点
// 参数: rect 旋转矩形
// 参数: func 回调函数
// 参数: xstep x坐标方向跳过点数
// 参数: ystep y坐标方向跳过点数
void for_each(const cv::RotatedRect &rect, const std::function<void(int, int)> &func, int xstep, int ystep)
{
	auto minimum = std::numeric_limits<float>::min();
	auto maximum = std::numeric_limits<float>::max();
  for_each(rect, minimum, minimum, maximum, maximum, func, xstep, ystep);
}


// 遍历旋转矩形内部的所有像素点
// 参数: rect 旋转矩形
// 参数: xmin x坐标最大值
// 参数: ymin y坐标最大值
// 参数: xmax x坐标最大值
// 参数: ymax y坐标最大值
// 参数: func 回调函数
// 参数: xstep x坐标方向跳过点数
// 参数: ystep y坐标方向跳过点数
void for_each(const cv::RotatedRect &rect, float xmin, float ymin, float xmax, float ymax, const std::function<void(int, int)> &func, int xstep, int ystep)
{
	// 获取矩形四个顶点的坐标
	const unsigned int sizeof_vertices = 4;
	cv::Point2f vertices[sizeof_vertices];
	rect.points(vertices);

	// 对坐标进行排序
	std::sort(std::begin(vertices), std::end(vertices), [](const cv::Point2f &point1, const cv::Point2f &point2)
	{
		return point1.y < point2.y;
	});

	// 最上面的点与距离其最近的点的一元一次方程系数和截距
  auto tsparam = get_linear_parameters<float>(vertices[0].x, vertices[0].y, vertices[1].x, vertices[1].y);

	// 最上面的点与距离其第二进的点的一元一次方程系数和截距
	auto tlparam = get_linear_parameters(vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y);

	// 最下面的点与距离其第二进的点的一元一次方程系数和截距
	auto blparam = get_linear_parameters(vertices[3].x, vertices[3].y, vertices[1].x, vertices[1].y);

	// 最下面的点与距离其最近的点的一元一次方程系数和截距
	auto bsparam = get_linear_parameters(vertices[3].x, vertices[3].y, vertices[2].x, vertices[2].y);

	// 正矩形
	if (std::abs(tsparam.first) < 0.0001 || std::abs(tlparam.first) < 0.0001)
	{
		auto data = std::minmax(vertices[0].x, vertices[1].x);
		xmin = std::max(data.first, xmin);
		xmax = std::min(data.second, xmax);
		ymin = std::max(vertices[0].y, ymin);
		ymax = std::min(vertices[2].y, ymax);
		for (float y = ymin; y < ymax; y += ystep)
		{
			for (float x = xmin; x < xmax; x += xstep)
			{
				func(static_cast<int>(x), static_cast<int>(y));
			}
		}
		return;
	}

	// 遍历上三角形
	auto yminimum = std::max(vertices[0].y, ymin);
	auto ymaximum = std::min(vertices[1].y, ymax);
	for (float y = yminimum; y < ymaximum; y += ystep)
	{
		auto x1 = get_linear_x(tsparam.first, tsparam.second, y);
		auto x2 = get_linear_x(tlparam.first, tlparam.second, y);
		auto data = std::minmax(x1, x2);

		auto xminimum = std::max(data.first, xmin);
		auto xmaximum = std::min(data.second, xmax);
		for (float x = xminimum; x < xmaximum; x += xstep)
		{
			func(static_cast<int>(x), static_cast<int>(y));
		}
	}

	// 遍历平行四边形
	yminimum = std::max(vertices[1].y, ymin);
	ymaximum = std::min(vertices[2].y, ymax);
	for (float y = yminimum; y < ymaximum; y += ystep)
	{
		auto x1 = get_linear_x(blparam.first, blparam.second, y);
		auto x2 = get_linear_x(tlparam.first, tlparam.second, y);
		auto data = std::minmax(x1, x2);

		auto xminimum = std::max(data.first, xmin);
		auto xmaximum = std::min(data.second, xmax);
		for (float x = xminimum; x < xmaximum; x += xstep)
		{
			func(static_cast<int>(x), static_cast<int>(y));
		}
	}

	// 遍历下三角形
	yminimum = std::max(vertices[2].y, ymin);
	ymaximum = std::min(vertices[3].y, ymax);
	for (float y = yminimum; y < ymaximum; y += ystep)
	{
		auto x1 = get_linear_x(blparam.first, blparam.second, y);
		auto x2 = get_linear_x(bsparam.first, bsparam.second, y);
		auto data = std::minmax(x1, x2);

		auto xminimum = std::max(data.first, xmin);
		auto xmaximum = std::min(data.second, xmax);
		for (float x = xminimum; x < xmaximum; x += xstep)
		{
			func(static_cast<int>(x), static_cast<int>(y));
		}
	}
}


// 计算直线的k、b
// 参数: x1 横坐标1, x2 横坐标2
// 参数: y1 纵坐标1, y2 纵坐标2
// 返回值: 直线的参数k、b
template <typename T>
std::pair<double, double> get_linear_parameters(T x1, T y1, T x2, T y2)
{
  if (x1 == x2)
  {
    return std::pair<double, double>(std::numeric_limits<long double>::max(), 0);
  }
  const auto k = (y1 - y2) / (x1 - x2);
  const auto b = y1 - k * x1;
  return std::pair<double, double>(k, b);
}

// 两点计算直线 k、b
// 参数: point1 点1
// 参数: point2 点2
// 返回值: 直线的参数 k、b
template <typename T>
std::pair<double, double> get_linear_parameters(T point1, T point2)
{
  if (point1.x == point2.x)
  {
    return std::pair<double, double>(std::numeric_limits<long double>::max(), 0);
  }
  const auto k = (point1.y - point2.y) / (point1.x - point2.x);
  const auto b = point1.y - k * point1.x;
  return std::pair<double, double>(k, b);
}

// 已知直线 k、b, 给定纵坐标 y 计算出坐标 x
// 参数: coefficient 斜率 k
// 参数: intercept 截距 b
// 返回值: 横坐标 x 的值
template <typename T>
T get_linear_x(double coefficient, double intercept, T y)
{
  if (coefficient == 0)
  {
    return 0;
  }
  return static_cast<T>((y - intercept) / coefficient);
}

// 已知直线 k、b, 给定横坐标 x 计算出坐标 y
// 参数: coefficient 斜率 k
// 参数: intercept 截距 b
// 返回值: 纵坐标 y 的值
template <typename T>
T get_linear_y(double coefficient, double intercept, T x)
{
  return static_cast<T>(coefficient * x + intercept);
}

// 计算两点距离的平方
// 参数: point1 第一个点
// 参数: point2 第二个点
// 返回值: 两点距离的平方
template <typename T>
double squaredist(const cv::Point_<T> &point1, const cv::Point_<T> &point2)
{
  return std::pow(point2.x - point1.x, 2) + std::pow(point2.y - point1.y, 2);
}

// 计算两点距离的平方
// 参数: x1 第一个点的x坐标
// 参数: y1 第一个点的y坐标
// 参数: x2 第二个点的x坐标
// 参数: y2 第二个点的y坐标
// 返回值: 两点距离的平方
template <typename T>
double squaredist(T x1, T y1, T x2, T y2)
{
  return std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2);
}

// 计算两点距离
// 参数: point1 第一个点
// 参数: point2 第二个点
// 返回值: 两点距离
template <typename T>
double distance(const cv::Point_<T> &point1, const cv::Point_<T> &point2)
{
  return std::sqrt(squaredist(point1, point2));
}

// 计算两点距离
// 参数: x1 第一个点的x坐标
// 参数: y1 第一个点的y坐标
// 参数: x2 第二个点的x坐标
// 参数: y2 第二个点的y坐标
// 返回值: 两点距离
template <typename T>
double distance(T x1, T y1, T x2, T y2)
{
  return std::sqrt(squaredist(x1, y1, x2, y2));
}

// 计算两点的中心点坐标
// 参数: point1 第一个点
// 参数: point2 第二个点
// 返回值: 两点的中心点坐标
template <typename T>
T center_point(const T &point1, const T &point2)
{
  return T((point1.x + point2.x) / 2, (point1.y + point2.y) / 2);
}

// 判断点与直线的距离
// 参数: coefficient 直线斜率 k
// 参数: intercept 直线截距 b
// 参数: point 需要计算的点
// 返回值: 点与直线的距离
template <typename T>
double point_to_line(double coefficient, double intercept, const T &point)
{
  return std::abs((coefficient * point.x - point.y + intercept) / std::sqrt(std::pow(coefficient, 2) + 1));
}

// 判断点是否在另外两点构成的直线上
// 参数: point1 直线点 1
// 参数: point2 直线点 2
// 参数: point 将要判断的点
// 返回值: true or false
template <typename T>
bool point_in_line(const T &point1, const T &point2, const T &point)
{
  auto hori = std::minmax(point1.x, point2.x);
  auto vert = std::minmax(point1.y, point2.y);
  if (point.x >= hori.first && point.x <= hori.second && point.y >= vert.first && point.y <= vert.second)
  {
    auto s1 = point.x - point1.x;
    auto s2 = point.y - point1.y;
    auto t1 = point1.x - point2.x;
    auto t2 = point1.y - point2.y;
    return std::abs(s1 * t2 - s2 * t1) < 0.01;
  }
  return false;
}
}
}
