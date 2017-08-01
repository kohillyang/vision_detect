#ifndef __DEBUG_UTILITY_H__
#define __DEBUG_UTILITY_H__

#ifdef _DEBUG_CONSOLE

#include <iostream>
#include <stdio.h>

#ifdef _DEBUG_TIME

#include <chrono>
#define speed_test_begin() speed_test_start_begin_time = std::chrono::system_clock::now()
#define speed_test_end(info, unit) std::cout << info << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - speed_test_start_begin_time).count() / 1000.0 << " " << unit << std::endl;

#ifdef _DEBUG_PART_TIME
#define speed_test_reset() speed_test_start_begin_time = std::chrono::system_clock::now()
#else // _DEBUG_PART_TIME
#define speed_test_reset()
#endif // _DEBUG_PART_TIME

#else // _DEBUG_TIME

#define speed_test_begin()
#define speed_test_reset()
#define speed_test_end(...)

#endif // _DEBUG_TIME


#define console_log(args...) printf(args)

#else // _DEBUG_CONSOLE

#define speed_test_begin()
#define speed_test_reset()
#define speed_test_end(...)
#define console_log(...)

#endif // _DEBUG_CONSOLE


#endif // !__DEBUG_UTILITY_H__
