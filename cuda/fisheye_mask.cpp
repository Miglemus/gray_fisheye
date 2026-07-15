#include <torch/extension.h>

torch::Tensor generate_fisheye_valid_mask_cuda(int model_id, torch::Tensor params, int height, int width,
                                               double radius_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_fisheye_valid_mask_cuda", &generate_fisheye_valid_mask_cuda);
}
