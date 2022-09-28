# FUME Image Translation

FUndamental Matrix based Epipolar Image Translation Layer.

## Development Setup

This repository is devided in two sections. The python part defines the pytorch 
interface implementing a `nn.module` and a `autograd.function`. This can be found 
in [fume_layer.py](fume_layer.py).

The underlying implementation of the image translation layer is found in the 
[cuda](cuda) folder. The `.cpp` files take care of framework-related functions
whilst the actual image transformations are implemented in 
[image_translation_kernel.cu](cuda/image_translation_kernel.cu). The 
[header file](cuda/image_translation.h) is only needed to build the test script in
[main.cpp](cuda/main.cpp) using the [CMakeLists.txt](cuda/CMakeLists.txt).


### Building the C++ test scripts 

The file [main.cpp](cuda/main.cpp) tests the image translation functionality 
without the python frontend. This is not needed to build the FUME layers. 
The only purpose of building this seperate C++ executable is to verify the 
functionality during development.

To set up your machine for development, follow these steps:

1. Download and unzip libtorch (I used [libtorch_linux_cuda11.3](https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip))
2. Copy path to folder (e.g. /usr/include/libtorch) and add it to the included directories in CMakeLists.txt
3. Add path to your python include directories. You can find out where that is by running `import sysconfig; print(sysconfig.get_paths()['include'])` in your preferred python env.
4. Make sure CUDA build tools are installed correctly. Verify by running `nvcc --version` (tested with 11.3). Also install CudNN. (tested with 7.6)
