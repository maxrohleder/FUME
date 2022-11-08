# FUME - Fundamental Matrix based Epipolar Image Translation (Pytorch)

This derivable pytorch operator allows to translate projection images to epipolar line images in another view. 
For a given image $p_1(u, v)$ with points $P_1$ and $P_2$ on it, this operator calculates and draws the epipolar lines $l_1$, $l_2$ in the consistency map $CM_2(u', v')$.

![Dual View Geometry](https://github.com/maxrohleder/FUME/blob/assets/img/DualViewLineGeometry.png)

When applied to entire images, epipolar consistency maps emerge. These maps can be used as a geometry informed prior during model training. Applications include the reduction of false positives in segmentation task and the improvement of segmentation of partially occluded objects using the second view.

After installation, run the [main.py](main.py) script to get this image:

![example image](https://github.com/maxrohleder/FUME/blob/assets/img/cubes_mapped.png).

#### NEW: Downsampled and padded Image translation

In Deep Learning models, it is often the case, that images are downsampled. The pre-calculated
fundamental matrices however require a fixed size (eg. `(976, 976)`). To enable a dynamic 
downsampling without having to instantiate a new layer per resolution, we introduced the
`downsampled_factor` parameter. 

So, for example, if you downsample an image by a factor of two and now your tensors 
spatial dimensions are `(488, 488)`, you can adapt this by setting this option.

```python
fume3d = Fume3dLayer()
factor = torch.tensor([downsample_factor], dtype=torch.float64, device='cuda', requires_grad=False)
CM1 = fume3d(view2_bin, F12, F21, downsampled_factor=factor)
CM2 = fume3d(view1_bin, F21, F12, downsampled_factor=factor)
```

Furthermore, the projection matrices need to be defined to map onto the center of the detector. Eg. if the 
detector has shape `(976, 976)`, the projection matrices need to be compensated by this:

```python
c = (976 / 2) - 0.5  # center of detector in pixels
to_center = np.array([[1, 0, -c],
                      [0, 1, -c],
                      [0, 0, 1]])
P1 = to_center @ P1
```

For more details see ![main.py lines 118ff](main.py).

This enables the user to get downsampled images like this:

![downsampling demo](https://github.com/maxrohleder/FUME/blob/assets/img/downsampling.png)

Padding and downsampling (needed in many CNN architectures) are also supported:

![padding demo](https://github.com/maxrohleder/FUME/blob/assets/img/Downsampled_Scaled.png)

## Install Instructions

First, install a suitable pytorch installation (https://pytorch.org/get-started/locally/). It comes with the necessary libraries build the cuda sources in this implementation. To install the latest version right from this repository, do this:

```shell
git clone https://github.com/maxrohleder/FUME.git
cd FUME
pip install -e .
```

To test the installation, run the main file (e.g. `python main.py`). You should get an image similar to the one above. The layers have been tested on Windows and linux using python 3.10 and the CUDA 11.3 installation of pytorch (comes with cudNN 8.0).

## Development Setup

This repository is devided in two sections. The python part defines the pytorch 
interface implementing a `nn.module` and a `autograd.function`. This can be found 
in [fume_layer.py](fume.py).

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
