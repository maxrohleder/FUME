#include <torch/extension.h>

#ifndef FUME_IMAGE_TRANSLATION_H
#define FUME_IMAGE_TRANSLATION_H

#endif //FUME_IMAGE_TRANSLATION_H

torch::Tensor translate_image(
        const torch::Tensor& input,
        const torch::Tensor& F,
        const torch::Tensor& factor);