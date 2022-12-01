cmake_minimum_required(VERSION 3.10)
project(MyProject)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(Example measure-speed.cpp Convolution.hpp cudnn.hpp cuda.hpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(cudart REQUIRED IMPORTED_TARGET cudart-10.2)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
target_link_libraries(Example PUBLIC opencv_imgcodecs PkgConfig::cudart cudnn)
target_include_directories(Example PUBLIC ../Time ../Statistics)

configure_file(input.png ${CMAKE_CURRENT_BINARY_DIR}/input.png COPYONLY)
