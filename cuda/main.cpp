#include <torch/torch.h>
#include <iostream>
#include "image_translation.h"

int main() {
    auto options = torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);
    torch::Tensor F = torch::tensor({{{  0.003976,   -0.944784,     469.808846},
                                                        { -0.950176,   -0.000314,   -3336.818453},
                                                        {462.822527, 4266.546518, -433140.893415}}}, options);

    torch::Tensor view1 = torch::zeros({1, 1, 976, 976}, options);
    torch::Tensor view2 = translate_image(view1, F);

    std::cout << F << std::endl;
    std::cout << view2 << std::endl;
}
