# to find this path, run this in your preferred python env: "import sysconfig;print(sysconfig.get_paths()['include'])"
include_directories(/home/dl/miniconda3/envs/fume/include/python3.10)
# set this to the directory where libtorch was unzipped to
set(CMAKE_PREFIX_PATH /usr/include/libtorch)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FUME CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(FUME main.cpp image_translation.h image_translation.cpp image_translation_kernel.cu)

set_target_properties(FUME PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(FUME "${TORCH_LIBRARIES}")
