cmake_minimum_required(VERSION 3.20)
project(EyeTracker)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(EyeTracker main.cpp)
target_link_libraries(EyeTracker ${OpenCV_LIBS})

