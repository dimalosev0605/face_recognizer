cmake_minimum_required(VERSION 3.5)

project(face_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_INCLUDE_DIR /home/dima/Boost/boost_1_72_0/Boost_Library_Binary/include)
set(Boost_LIBRARY_DIR /home/dima/Boost/boost_1_72_0/Boost_Library_Binary/lib)

find_package( OpenCV REQUIRED )
find_package(dlib REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /home/dima/opencv-4.2.0/opencv_contrib-master/modules/face/include )
include_directories(${Boost_INCLUDE_DIR})

link_directories(${Boost_LIBRARY_DIR})
add_executable(face_recognition main.cpp)
target_link_libraries(  face_recognition ${OpenCV_LIBS} /home/dima/opencv-4.2.0/build/lib/libopencv_face.so dlib::dlib
                        boost_system boost_filesystem
                     )
