cmake_minimum_required(VERSION 2.8.4)
project(jacobi2_0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(jacobi2_0 ${SOURCE_FILES})