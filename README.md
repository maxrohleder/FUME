# FUME Image Translation

FUndamental Matrix based Epipolar Image Translation Layer.

## Development Setup

1. Download and unzip libtorch (libtorch_cuda11.3)[https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip]
2. Copy path to folder (e.g. /usr/include/libtorch and paste it in CMakeLists.txt line 2)

### CUDA compiler and CMAKE
```
-- The CUDA compiler identification is unknown
CMake Error at C:/Program Files/JetBrains/CLion 2022.2.3/bin/cmake/win/share/cmake-3.23/Modules/CMakeDetermineCUDACompiler.cmake:633 (message):
Failed to detect a default CUDA architecture.
```

- Add LD_LIBARY_PATH and PATH to your Cuda installation
- make sure the version specified ```set(CMAKE_CUDA_ARCHITECTURES 75 86)``` makes sense for your GPU
- make sure to link