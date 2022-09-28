#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


// CUDA kernel implementation
template <typename scalar_t>
__global__ void translate_image_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> F,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output) {

    // indexing the batches and each spatial dimensions of input (batch, v, u)
    auto b = blockIdx.x * blockDim.x + threadIdx.x;
    auto v = blockIdx.y * blockDim.y + threadIdx.y;
    auto u = blockIdx.z * blockDim.z + threadIdx.z;

    // using accessors, we can use image.size(idx) and indexing image[c][r]
    for (int c = 0; c < image.size(1); c++){
        output[b][c][v][u] = 1;
    }
}


/**
 * Calls the cuda kernel above to calculate the epipolar image of the given input image. This is done using the
 * fundamental matrix F which maps points in the output view onto lines in the input image. These lines are
 * integrated over.
 * @param input image data from view1 in shape (B, C, H, W)
 * @param F Fundamental Matrix in shape (B, 3, 3) to translate a coordinate x2 in view2 to a line l1 in view1
 * @return translated view containing epipolar lines in same shape as input data
 */
torch::Tensor translate_image_cuda( const torch::Tensor& input, const torch::Tensor& F) {

    // define constants and output
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    auto new_image = torch::zeros_like(input);

    // define a 3d grid. 4 in batch dimension, 16 in x and y
    dim3 threadsPerBlock(4,16, 16);  // b, v, u
    dim3 numBlocks((batch_size + threadsPerBlock.x -1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y -1) / threadsPerBlock.y,
                   (width + threadsPerBlock.z -1) / threadsPerBlock.z);

    // invoke kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "translate_image_cuda", [&] {
        translate_image_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            F.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            new_image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    });

    return new_image;
}
