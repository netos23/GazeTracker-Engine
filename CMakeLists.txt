cmake_minimum_required(VERSION 3.17)
project(EyeTracker)

set(CMAKE_CXX_STANDARD 11)
include(CMakeListsPrivate.txt)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(EyeTracker main.cpp)
target_link_libraries(EyeTracker ${OpenCV_LIBS})

