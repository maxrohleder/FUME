#include <torch/extension.h>

// CUDA forward declarations (implemented in image_translation_kernel.cu)

torch::Tensor translate_image_cuda(
        const torch::Tensor& input,
        const torch::Tensor& F) ;

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor translate_image(
    const torch::Tensor& input,
    const torch::Tensor& F) {
  CHECK_INPUT(input);
  CHECK_INPUT(F);

  return translate_image_cuda(input, F);
}
