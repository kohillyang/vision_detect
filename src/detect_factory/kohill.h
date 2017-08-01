/*
 * print.h
 *
 *  Created on: Jul 25, 2017
 *      Author: kohill
 */

#ifndef SRC_DETECT_FACTORY_KOHILL_H_
#define SRC_DETECT_FACTORY_KOHILL_H_
namespace kohill{
#include <iostream>
template<typename T>
void print(T x){
//	std::cout << x << std::endl;
}
template<typename T0,typename T1>
void print(T0 x,T1 y){
//	std::cout << x <<" "<<y<< std::endl;
}
template<typename T0,typename T1,typename T2>
void print(T0 x,T1 y,T2 z){
//	std::cout << x <<" "<<y<< ""<< z<<std::endl;
}
};




#endif /* SRC_DETECT_FACTORY_KOHILL_H_ */
