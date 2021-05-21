cmake_minimum_required(VERSION 3.10)
project(camera-calibration)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(single-camera-calibration single-camera-calibration/main.cpp single-camera-calibration/OpenCVCalibration.cpp CalibUtils.cpp)
target_link_libraries(single-camera-calibration ${OpenCV_LIBS})

add_executable(stereo-camera-calibration stereo-camera-calibration/main.cpp stereo-camera-calibration/OpenCVStereoCalibration.cpp  single-camera-calibration/OpenCVCalibration.cpp CalibUtils.cpp)
target_link_libraries(stereo-camera-calibration ${OpenCV_LIBS})