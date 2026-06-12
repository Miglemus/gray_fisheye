#pragma once

#ifdef __CUDACC__
#include "../utils/random.h"
#include "../utils/vec_math.h"
#include "opencv_fisheye.cuh"
#include "tpf.cuh"
#endif

// * Primary-ray camera models
#define CAMERA_MODEL_PINHOLE 0
#define CAMERA_MODEL_OPENCV_FISHEYE 1
#define CAMERA_MODEL_THIN_PRISM_FISHEYE 2

struct Camera {
    const float3 *origin;
    const float *vertical_fov_radians;
    const float3 *rotation_c2w; // * stores 3 rows for each 3x3 matrix
    const float3 *rotation_w2c; // * stores 3 rows for each 3x3 matrix
    const float *znear;
    const float *zfar;

    const int *model_id;          // * CAMERA_MODEL_* selector
    const float *fisheye_params;  // * up to 12 floats: fx, fy, cx, cy, k1..k4 [, p1, p2, sx1, sy1]

#ifdef __CUDACC__
    __device__ float3 compute_primary_ray_direction(const bool jitter, const uint3 idx, const uint3 dim,
                                                    unsigned int &seed) {
        // * Compute sub-pixel jitter
        float2 idxf = make_float2(idx.x, idx.y);
        if (jitter) {
            const float2 jitter_offset = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
            idxf += jitter_offset;
        }

        if (*model_id == CAMERA_MODEL_OPENCV_FISHEYE) {
            // * Unproject pixel to a unit bearing in the OpenCV camera frame (x right, y down, z fwd)
            float3 ocv = opencv_fisheye_unproject(fisheye_params, idxf.x + 0.5f, idxf.y + 0.5f);
            if (ocv.x == 0.0f && ocv.y == 0.0f && ocv.z == 0.0f) {
                return make_float3(0.0f, 0.0f, 0.0f); // * Inactive pixel (outside the fisheye FOV)
            }
            // * Convert to GRay's camera frame (x right, y up, z back) then rotate to world
            float3 cam_dir = make_float3(ocv.x, -ocv.y, -ocv.z);
            return normalize(rotation_w2c[0] * cam_dir.x + rotation_w2c[1] * cam_dir.y +
                             rotation_w2c[2] * cam_dir.z);
        } else if (*model_id == CAMERA_MODEL_THIN_PRISM_FISHEYE) {
            // * Unproject pixel to a unit bearing in the Thin Prism Fisheye camera frame (x right, y down, z fwd)
            float3 tpf = thin_prism_fisheye_unproject(fisheye_params, idxf.x + 0.5f, idxf.y + 0.5f);
            if (tpf.x == 0.0f && tpf.y == 0.0f && tpf.z == 0.0f) {
                return make_float3(0.0f, 0.0f, 0.0f); // * Inactive pixel (outside the Thin Prism Fisheye FOV)
            }
            // * Convert to GRay's camera frame (x right, y up, z back) then rotate to world
            float3 cam_dir = make_float3(tpf.x, -tpf.y, -tpf.z);
            return normalize(rotation_w2c[0] * cam_dir.x + rotation_w2c[1] * cam_dir.y +
                             rotation_w2c[2] * cam_dir.z);
        }

        // * Pinhole: NDC image-plane coordinates (x right, y up, forward = -z)
        float view_size = tan(*vertical_fov_radians / 2);
        float aspect_ratio = float(dim.x) / float(dim.y);
        float y = view_size * (1.0f - 2.0f * (idxf.y + 0.5f) / (float(dim.y)));
        float x = aspect_ratio * view_size * (2.0f * (idxf.x + 0.5f) / (float(dim.x)) - 1.0f);

        // * Rotate to world and normalize (n.b. multiplies by *transposed* w2c)
        return normalize(rotation_w2c[0] * x + rotation_w2c[1] * y - rotation_w2c[2]);
    }
#endif
};

#ifndef __CUDACC__
#include "headers.h"

struct CameraDataHolder : torch::CustomClassHolder {
    Tensor origin = torch::zeros({3}, CUDA_FLOAT32);
    Tensor vertical_fov_radians = torch::zeros({1}, CUDA_FLOAT32);
    Tensor rotation_c2w = torch::zeros({3, 3}, CUDA_FLOAT32);
    Tensor rotation_w2c = torch::zeros({3, 3}, CUDA_FLOAT32);
    Tensor znear = torch::zeros({1}, CUDA_FLOAT32);
    Tensor zfar = torch::zeros({1}, CUDA_FLOAT32);

    Tensor model_id = torch::zeros({1}, CUDA_INT32);        // * defaults to CAMERA_MODEL_PINHOLE
    Tensor fisheye_params = torch::zeros({12}, CUDA_FLOAT32); // * fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1

    Camera reify() {
        return Camera{
            .origin = reinterpret_cast<float3 *>(origin.data_ptr()),
            .vertical_fov_radians = reinterpret_cast<float *>(vertical_fov_radians.data_ptr()),
            .rotation_c2w = reinterpret_cast<float3 *>(rotation_c2w.data_ptr()),
            .rotation_w2c = reinterpret_cast<float3 *>(rotation_w2c.data_ptr()),
            .znear = reinterpret_cast<float *>(znear.data_ptr()),
            .zfar = reinterpret_cast<float *>(zfar.data_ptr()),
            .model_id = reinterpret_cast<int *>(model_id.data_ptr()),
            .fisheye_params = reinterpret_cast<float *>(fisheye_params.data_ptr()),
        };
    }

    void set_pose(const Tensor &c2w_origin, const Tensor &c2w_rotation) {
        TORCH_CHECK(c2w_rotation.sizes() == torch::IntArrayRef({3, 3}), "c2w_rotation must be 3x3");
        TORCH_CHECK(c2w_origin.sizes() == torch::IntArrayRef({3}), "c2w_origin must be 3");
        rotation_c2w.copy_(c2w_rotation);
        origin.copy_(c2w_origin);
        rotation_w2c.copy_(c2w_rotation.transpose(0, 1));
    }

    void set_pinhole() { model_id.fill_(CAMERA_MODEL_PINHOLE); }

    void set_opencv_fisheye(const Tensor &params) {
        TORCH_CHECK(params.numel() == 8, "fisheye params must have 8 elements (fx, fy, cx, cy, k1..k4)");
        model_id.fill_(CAMERA_MODEL_OPENCV_FISHEYE);
        fisheye_params.slice(0, 0, 8).copy_(params.reshape({8}));
    }

    void set_thin_prism_fisheye(const Tensor &params) {
        TORCH_CHECK(params.numel() == 12, "thin prism fisheye params must have 12 elements (fx, fy, cx, cy, k1..k4, p1, p2, sx1, sy1)");
        model_id.fill_(CAMERA_MODEL_THIN_PRISM_FISHEYE);
        fisheye_params.copy_(params.reshape({12}));
    }

    static void bind(torch::Library &m) {
        m.class_<CameraDataHolder>("CameraDataHolder")
            .def("set_pose", &CameraDataHolder::set_pose)
            .def("set_pinhole", &CameraDataHolder::set_pinhole)
            .def("set_opencv_fisheye", &CameraDataHolder::set_opencv_fisheye)
            .def("set_thin_prism_fisheye", &CameraDataHolder::set_thin_prism_fisheye)
            .def_readonly("vertical_fov_radians", &CameraDataHolder::vertical_fov_radians)
            .def_readonly("znear", &CameraDataHolder::znear)
            .def_readonly("zfar", &CameraDataHolder::zfar);
    }
};
#endif
