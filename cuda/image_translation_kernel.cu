#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// CUDA kernel implementation

template <typename scalar_t>
__global__ void translate_image_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> F,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output) {
    // column index
    const int c = blockIdx.y;
    // row index
    const int r = blockIdx.x * blockDim.x + threadIdx.x;

    // using accessors, we can use image.size(idx) and indexing image[c][r]
    // TODO implement functionality
    output[c][r] = 1
}


// invoking the kernel for a given image

std::vector<torch::Tensor> translate_image_cuda( torch::Tensor input, torch::Tensor F) {

    // define constants and output
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    auto new_image = torch::zeros_like(input);

    // TODO define a 2d grid
    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    // invoke kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "translate_image_cuda", ([&] {
    translate_image_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        F.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    return {new_image};
}
