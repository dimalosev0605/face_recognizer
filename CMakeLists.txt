cmake_minimum_required(VERSION 3.5)

project(face_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_CONTRIB_INCLIDE_DIRS "/home/dima/opencv-4.2.0/opencv_contrib-master/modules/face/include")
set(OpenCV_CONTRIB_FACE_LIBRARY "/home/dima/opencv-4.2.0/build/lib/libopencv_face.so")

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.72.0 REQUIRED filesystem)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_CONTRIB_INCLIDE_DIRS})

add_executable(face_recognition main.cpp)
target_link_libraries(face_recognition dlib::dlib ${OpenCV_LIBS} ${Boost_LIBRARIES} ${OpenCV_CONTRIB_FACE_LIBRARY})

