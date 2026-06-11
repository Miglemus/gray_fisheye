#pragma once

#ifdef __CUDACC__
#include "../utils/vec_math.h"

// * Inverse of the COLMAP/OpenCV THIN_PRISM_FISHEYE projection.
// * Given a pixel (u, v) and intrinsics (fx, fy, cx, cy, k1..k4), returns the unit
// * bearing in the Thin Prism Fisheye camera frame (x right, y down, z forward).
// * Returns (0, 0, 0) for pixels outside the valid field of view (theta >= 90 deg),
// * i.e. the black vignette corners of a circular Thin Prism Fisheye; the caller treats a
// * zero-length direction as an inactive pixel that renders the background.
__device__ __forceinline__ float3 thin_prism_fisheye_unproject(const float *params, float u, float v) {
    const float fx = params[0];
    const float fy = params[1];
    const float cx = params[2];
    const float cy = params[3];
    const float k1 = params[4];
    const float k2 = params[5];
    const float k3 = params[6];
    const float k4 = params[7];
    const float p1 = params[8];
    const float p2 = params[9];
    const float sx1 = params[10];
    const float sy1 = params[11];

    // Coordonnées cibles normalisées distordues
    float u_target = (u - cx) / fx;
    float v_target = (v - cy) / fy;

    // Initialisation du point fixe (en float)
    float u_final = u_target;
    float v_final = v_target;

    const int MAX_ITERATIONS = 20;
    const float EPSILON = 1e-6f; // Précision idéale pour du float32

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        float r2 = u_final * u_final + v_final * v_final;
        float r = sqrtf(r2);
        
        float u_fe = u_final;
        float v_fe = v_final;
        
        if (r > 0.0f) {
            float theta = atanf(r);
            float theta2 = theta * theta;
            float theta4 = theta2 * theta2;
            float theta6 = theta4 * theta2;
            float theta8 = theta4 * theta4;
            
            float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            u_fe = (theta_d / r) * u_final;
            v_fe = (theta_d / r) * v_final;
        }
        
        float u_estimated = u_fe + 2.0f * p1 * u_final * v_final + p2 * (r2 + 2.0f * u_final * u_final) + sx1 * r2;
        float v_estimated = v_fe + p1 * (r2 + 2.0f * v_final * v_final) + 2.0f * p2 * u_final * v_final + sy1 * r2;

        // Calcul du résidu (Erreur)
        float delta_u = u_target - u_estimated;
        float delta_v = v_target - v_estimated;
        
        // Correction de l'estimé
        u_final += delta_u;
        v_final += delta_v;
        
        // Arrêt précoce dès que la précision flotteur est atteinte
        if ((delta_u * delta_u + delta_v * delta_v) < EPSILON) {
            break;
        }
    }

    // * FOV cutoff at 90 deg (matches opencv_fisheye.cuh and fisheye_mask.py)
    constexpr float FISHEYE_MAX_THETA = 1.5707963f;
    float r_final = sqrtf(u_final * u_final + v_final * v_final);
    float theta_final = atanf(r_final);
    float t2 = FISHEYE_MAX_THETA * FISHEYE_MAX_THETA;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;
    float theta_d_max = FISHEYE_MAX_THETA * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
    float theta2_f = theta_final * theta_final;
    float theta4_f = theta2_f * theta2_f;
    float theta6_f = theta4_f * theta2_f;
    float theta8_f = theta4_f * theta4_f;
    float theta_d_final =
        theta_final * (1.0f + k1 * theta2_f + k2 * theta4_f + k3 * theta6_f + k4 * theta8_f);
    if (theta_final >= FISHEYE_MAX_THETA || theta_d_final >= theta_d_max) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float inv_norm = rsqrtf(u_final * u_final + v_final * v_final + 1.0f);
    return make_float3(u_final * inv_norm, v_final * inv_norm, 1.0f * inv_norm);
}
#endif