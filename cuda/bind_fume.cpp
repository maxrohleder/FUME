//
// This file is only needed to bind the cpp project to a python library with setuptools. See setup.py.
//
#include "image_translation.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("translate", &translate_image, "Epipolar Image Translation (CUDA)");
}
