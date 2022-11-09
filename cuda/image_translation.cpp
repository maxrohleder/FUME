#include <torch/extension.h>

// CUDA forward declarations (implemented in image_translation_kernel.cu)
torch::Tensor translate_image_cuda(
        const torch::Tensor& input,
        const torch::Tensor& F,
        const torch::Tensor& factor) ;

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor translate_image(
    const torch::Tensor& input,
    const torch::Tensor& F,
    const torch::Tensor& factor) {
//  CHECK_INPUT(input);  this failed in the backward pass with batchsize > 1 09.11.2022 maybe investigate why.
//  CHECK_INPUT(F);

  return translate_image_cuda(input, F, factor);
}
