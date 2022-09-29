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
    auto u = blockIdx.y * blockDim.y + threadIdx.y;
    auto v = blockIdx.z * blockDim.z + threadIdx.z;

    // check boundaries
    if (b < output.size(0) && u < output.size(2) && v < output.size(3)) {

        // using accessors, we can use image.size(idx) and indexing image[x][y]
        for (int c = 0; c < output.size(1); c++) {

            // calculate homogenous epipolar line ax + by + c = 0
            scalar_t l1 = F[b][0][0] * u + F[b][0][1] * v + F[b][0][2];
            scalar_t l2 = F[b][1][0] * u + F[b][1][1] * v + F[b][1][2];
            scalar_t l3 = F[b][2][0] * u + F[b][2][1] * v + F[b][2][2];

            // calculate line equation parameters y = mx + t aka. v = mu + t
            scalar_t m = - l1 / l2;
            scalar_t t = - l3 / l2;

            // sum over line in input image
            scalar_t res = 0;

            for (int u1 = 0; u1 < image.size(2); u1++) {

                // line equation
                scalar_t v1 = m * u1 + t;

                // check boundaries in input image
                if (v1 >= image.size(3) | v1 < 0 | isnan(v1)) {
                    continue;
                }

                // calculate interpolation coefficients
                int v1c = (int) ceil(v1);
                scalar_t cdist = abs(ceil(v1) - v1);
                int v1f = (int) floor(v1);
                scalar_t fdist = abs(floor(v1) - v1);

                // add interpolated values
                res += cdist * image[b][c][u1][v1f] + fdist * image[b][c][u1][v1c];
            }
            // assign value of line-integral
            output[b][c][u][v] = res;
        }
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
    const int channels = (int) input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    auto new_image = torch::zeros_like(input);

    // define a 3d grid. 4 in batch dimension, 16 in x and y
    dim3 threadsPerBlock(4, 16, 16);  // b, v, u
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
