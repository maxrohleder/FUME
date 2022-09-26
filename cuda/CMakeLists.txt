# set this to the directory where libtorch was unzipped to
set(CMAKE_PREFIX_PATH /usr/include/libtorch)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FUME CXX)
set(CMAKE_CUDA_STANDARD 14)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(FUME main.cu)

set_target_properties(FUME PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(FUME "${TORCH_LIBRARIES}")
