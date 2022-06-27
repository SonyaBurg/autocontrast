cmake_minimum_required(VERSION 3.21)
project(evm)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-fopenmp -O3 -mfma")

add_executable(evm main.cpp)
