cmake_minimum_required(VERSION 2.8.3)
project(vision_unit)
set(CMAKE_BUILD_TYPE,Debug)
add_definitions(-std=c++11)
set(BOOST_ROOT "/opt/boost")

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    sensor_msgs
    roscpp
    tf
    move_base_msgs
    geometry_msgs
    message_generation    
    )

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)

add_message_files(
    FILES
    armor_msg.msg
    transformed_scan.msg
    )

## Generate services in the 'srv' folder
add_service_files(
    FILES
    SetGoal.srv
    )

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    move_base_msgs
    geometry_msgs
    )


# include_directories(include)
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Caffe_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    /home/kohill/tx1/backup/caffe-master/include
    )

include_directories(
    include/vision_unit
    include/vision_unit/detect_factory
    include/vision_unit/vision_lidar_fusion/
    /usr/include/python2.7
    )

add_executable(armor_detect
    src/detect_factory/armor_detect.cpp
    src/detect_factory/draw.cpp
    src/detect_factory/image_utility.cpp
    src/detect_factory/main.cpp
    src/detect_factory/labeler.cpp
    src/detect_factory/armor_detect_node.cpp
    src/detect_factory/util.cpp
    src/detect_factory/video_recoder.cpp
    src/detect_factory/circle_detect.cpp
    src/detect_factory/rectang_det_HQG.cpp
    )
add_dependencies(armor_detect ${catkin_EXPORTED_TARGETS})
	target_link_libraries(armor_detect 
	${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} 
	/usr/lib/arm-linux-gnueabihf/libpython2.7.so
)

