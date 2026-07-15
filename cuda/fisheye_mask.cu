#include <torch/extension.h>
#include "core/opencv_fisheye.cuh"
#include "core/rtpf.cuh"
#include "core/tpf.cuh"

#define CAMERA_MODEL_OPENCV_FISHEYE 1
#define CAMERA_MODEL_THIN_PRISM_FISHEYE 2
#define CAMERA_MODEL_RAD_TAN_THIN_PRISM_FISHEYE 3

__device__ __forceinline__ float opencv_fisheye_theta_d_at_max(const float *params) {
    const float k1 = params[4];
    const float k2 = params[5];
    const float k3 = params[6];
    const float k4 = params[7];
    constexpr float t = 1.5707963f;
    const float t2 = t * t;
    const float t4 = t2 * t2;
    const float t6 = t4 * t2;
    const float t8 = t4 * t4;
    return t * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
}

__device__ __forceinline__ bool opencv_fisheye_pixel_valid(const float *params, const float x, const float y,
                                                           const float radius_scale) {
    const float fx = params[0];
    const float fy = params[1];
    const float cx = params[2];
    const float cy = params[3];

    const float3 bearing = opencv_fisheye_unproject(params, x, y);
    if (bearing.x == 0.0f && bearing.y == 0.0f && bearing.z == 0.0f) {
        return false;
    }

    if (radius_scale >= 1.0f) {
        return true;
    }

    const float xd = (x - cx) / fx;
    const float yd = (y - cy) / fy;
    const float theta_d = sqrtf(xd * xd + yd * yd);
    return theta_d < radius_scale * opencv_fisheye_theta_d_at_max(params);
}

__device__ __forceinline__ bool thin_prism_fisheye_pixel_valid(const float *params, const float x, const float y,
                                                               const float radius_scale) {
    const float3 bearing = thin_prism_fisheye_unproject(params, x, y);
    if (bearing.x == 0.0f && bearing.y == 0.0f && bearing.z == 0.0f) {
        return false;
    }

    if (radius_scale >= 1.0f) {
        return true;
    }

    const float fx = params[0];
    const float fy = params[1];
    const float cx = params[2];
    const float cy = params[3];
    const float *extra_params = params + 4;

    float uu = (x - cx) / fx;
    float vv = (y - cy) / fy;
    const float uu0 = uu;
    const float vv0 = vv;

    constexpr int kMaxIterations = 100;
    constexpr float kMinStepSq = 1e-10f;
    bool converged = false;
    for (int i = 0; i < kMaxIterations; ++i) {
        float du, dv;
        thin_prism_fisheye_distortion(extra_params, uu, vv, &du, &dv);
        const float next_uu = uu0 - du;
        const float next_vv = vv0 - dv;
        const float step_u = next_uu - uu;
        const float step_v = next_vv - vv;
        uu = next_uu;
        vv = next_vv;
        if (step_u * step_u + step_v * step_v < kMinStepSq) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        return false;
    }

    const float theta = sqrtf(uu * uu + vv * vv);
    return theta < radius_scale * 1.5707963f;
}

__device__ __forceinline__ bool rad_tan_thin_prism_fisheye_pixel_valid(const float *params, const float x,
                                                                       const float y, const float radius_scale) {
    const float3 bearing = rad_tan_thin_prism_fisheye_unproject(params, x, y);
    if (bearing.x == 0.0f && bearing.y == 0.0f && bearing.z == 0.0f) {
        return false;
    }

    if (radius_scale >= 1.0f) {
        return true;
    }

    const float fx = params[0];
    const float fy = params[1];
    const float cx = params[2];
    const float cy = params[3];
    const float *extra_params = params + 4;

    float uu = (x - cx) / fx;
    float vv = (y - cy) / fy;
    const float uu0 = uu;
    const float vv0 = vv;

    constexpr int kMaxIterations = 100;
    constexpr float kMinStepSq = 1e-10f;
    bool converged = false;
    for (int i = 0; i < kMaxIterations; ++i) {
        float du, dv;
        rad_tan_thin_prism_fisheye_distortion(extra_params, uu, vv, &du, &dv);
        const float next_uu = uu0 - du;
        const float next_vv = vv0 - dv;
        const float step_u = next_uu - uu;
        const float step_v = next_vv - vv;
        uu = next_uu;
        vv = next_vv;
        if (step_u * step_u + step_v * step_v < kMinStepSq) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        return false;
    }

    const float theta = sqrtf(uu * uu + vv * vv);
    return theta < radius_scale * 1.5707963f;
}

__device__ __forceinline__ bool fisheye_pixel_valid(const int model_id, const float *params, const float x,
                                                    const float y, const float radius_scale) {
    if (model_id == CAMERA_MODEL_OPENCV_FISHEYE) {
        return opencv_fisheye_pixel_valid(params, x, y, radius_scale);
    }
    if (model_id == CAMERA_MODEL_THIN_PRISM_FISHEYE) {
        return thin_prism_fisheye_pixel_valid(params, x, y, radius_scale);
    }
    if (model_id == CAMERA_MODEL_RAD_TAN_THIN_PRISM_FISHEYE) {
        return rad_tan_thin_prism_fisheye_pixel_valid(params, x, y, radius_scale);
    }
    return false;
}

__global__ void fisheye_valid_mask_kernel(const int model_id, const float *params, const int height, const int width,
                                          const float radius_scale, bool *mask) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const bool valid = fisheye_pixel_valid(model_id, params, static_cast<float>(x) + 0.5f,
                                           static_cast<float>(y) + 0.5f, radius_scale);
    mask[y * width + x] = valid;
}

torch::Tensor generate_fisheye_valid_mask_cuda(const int model_id, torch::Tensor params, const int height,
                                               const int width, const double radius_scale) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(model_id >= CAMERA_MODEL_OPENCV_FISHEYE &&
                    model_id <= CAMERA_MODEL_RAD_TAN_THIN_PRISM_FISHEYE,
                "unsupported fisheye model id");

    const int num_params = (model_id == CAMERA_MODEL_OPENCV_FISHEYE)      ? 8
                           : (model_id == CAMERA_MODEL_THIN_PRISM_FISHEYE) ? 12
                                                                             : 16;
    TORCH_CHECK(params.numel() == num_params, "fisheye params have unexpected size");

    auto options = torch::TensorOptions().dtype(torch::kBool).device(params.device());
    torch::Tensor mask = torch::empty({height, width}, options);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fisheye_valid_mask_kernel<<<grid, block>>>(model_id, params.data_ptr<float>(), height, width,
                                               static_cast<float>(radius_scale), mask.data_ptr<bool>());
    cudaDeviceSynchronize();

    return mask;
}
