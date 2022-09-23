set(CMAKE_CUDA_COMPILER "C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v11.2\\bin\\nvcc.exe")

cmake_minimum_required(VERSION 3.23)
project(FUME CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(FUME main.cpp)

set_target_properties(FUME PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
