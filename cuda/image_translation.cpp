#include <torch/extension.h>
#include <vector>

// CUDA forward declarations (implemented in image_translation_kernel.cu)

std::vector<torch::Tensor> translate_image_cuda(
    torch::Tensor input,
    torch::Tensor F);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> translate_image(
    torch::Tensor input,
    torch::Tensor F) {
  CHECK_INPUT(input);
  CHECK_INPUT(F);

  return translate_image_cuda(input, F);
}

// Python bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("translate", &translate_image, "Epipolar Image Translation (CUDA)");
}
