cmake_minimum_required(VERSION 3.20)
project(EyeTracker)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} dlib::dlib)
add_executable(EyeTracker
        main.cpp GazeTracker.cpp GazeTracker.h math_tools.cpp math_tools.h FaceTracker.cpp FaceTracker.h Trackers.h)
target_link_libraries(EyeTracker ${OpenCV_LIBS} dlib::dlib)

