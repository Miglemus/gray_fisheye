"""Where does the rttpf rung's inverse disagree with gray's raygen? (debug scratch)

Splits the discrepancy into (a) the base solver's own consistency, (b) convergence of the
quasi-Newton, (c) a genuine model mismatch, by redoing the solve in float64.
"""

import dataclasses
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/gray/worktrees/rttpf-intrinsics")
sys.path.insert(0, "/workspace/gray/worktrees/rttpf-intrinsics/tests")

from test_camera_model import HEIGHT, WIDTH, build_scene, fisheye_camera  # noqa: E402

from gray.camera_model import rttpf_plate_scales, rttpf_project  # noqa: E402
from gray.raytracer import Raytracer  # noqa: E402

raytracer = build_scene("rttpf")
camera = dataclasses.replace(fisheye_camera(), R=np.eye(3))
base = raytracer.base_bearings(camera)
table = raytracer.camera_model.rttpf_table(base)
valid_base = base["bearings"].norm(dim=-1) > 0.5

# --- (a) is the base bearing consistent with the pixel it is supposed to come from?
target = table["target"]
grid_y, grid_x = torch.meshgrid(
    torch.arange(HEIGHT, device="cuda", dtype=torch.float32),
    torch.arange(WIDTH, device="cuda", dtype=torch.float32),
    indexing="ij",
)
for offset in (0.0, 0.5):
    pixels = torch.stack([grid_x + offset, grid_y + offset], dim=-1)
    error = (target - pixels).norm(dim=-1)[valid_base]
    print(f"base consistency, pixel offset {offset}: max {error.max():.3e} px  "
          f"median {error.median():.3e} px")

deltas = np.zeros(16)
deltas[0], deltas[1] = 0.004 * table["params"][0].item(), -0.003 * table["params"][1].item()
deltas[2], deltas[3] = 1.5, -0.8
deltas[4], deltas[5], deltas[6] = 2.0e-3, -8.0e-4, 3.0e-4
deltas[10], deltas[11] = 5.0e-4, -4.0e-4
deltas[12], deltas[14] = 1.2e-3, -9.0e-4
perturbed_params = table["params"] + torch.tensor(deltas, dtype=torch.float32, device="cuda")

# --- (b) an exact float64 solve of the SAME problem: project(w, perturbed) == target
w = table["w0"].to(torch.float64).clone()
params64 = perturbed_params.to(torch.float64)
target64 = target.to(torch.float64)
inv_focal = torch.stack([1.0 / params64[0], 1.0 / params64[1]])
for iteration in range(40):
    unit, inv_slope, inv_growth = rttpf_plate_scales(w, params64)
    residual = (rttpf_project(w, params64) - target64) * inv_focal
    radial = (residual * unit).sum(-1)
    step = unit * (radial * inv_slope).unsqueeze(-1) + (
        residual - radial.unsqueeze(-1) * unit
    ) * inv_growth.unsqueeze(-1)
    w = w - step
    if iteration in (0, 1, 2, 3, 5, 10, 39):
        left = (rttpf_project(w, params64) - target64).norm(dim=-1)[valid_base].max()
        print(f"float64 newton, step {iteration + 1:2d}: residual {left:.3e} px")

# --- what the rung actually produces, and what the raygen says
from gray.camera_model import rttpf_solve  # noqa: E402

lens = raytracer.camera_model.lens(camera.uid)
focal = torch.tensor([table["params"][0], table["params"][1]] * 2, device="cuda")
coefficients = torch.as_tensor(deltas, dtype=torch.float32, device="cuda").clone()
coefficients[:4] /= focal
coefficients = coefficients * table["scales"]
with torch.no_grad():
    lens.intrinsics.copy_(coefficients)
    delta_theta, delta_phi, residual_px = rttpf_solve(lens.intrinsics, table)
print(f"float32 rung: reported worst-pixel residual {residual_px.item():.3e} px")

reference = raytracer.base_bearings(dataclasses.replace(camera, intrinsics=camera.intrinsics + deltas))
valid = valid_base & (reference["bearings"].norm(dim=-1) > 0.5)
print(f"valid pixels: {int(valid.sum())} of {valid.numel()}")

# The float64 solution as a bearing, to compare against the raygen without any of the
# rung's rotation machinery in the way.
theta = w.norm(dim=-1)
scale = torch.where(theta > 1e-12, torch.tan(theta) / theta.clamp_min(1e-12), torch.ones_like(theta))
exact = torch.stack([w[..., 0] * scale, w[..., 1] * scale, torch.ones_like(theta)], dim=-1)
exact = exact / exact.norm(dim=-1, keepdim=True)

with torch.no_grad():
    _, direction = raytracer.camera_model(
        camera,
        {**base, "rotation": Raytracer._rotation_c2w_cuda(camera), "origin": camera.origin_cuda()},
        HEIGHT,
        WIDTH,
    )
moved = (reference["bearings"] - base["bearings"]).norm(dim=-1)[valid]
print(f"perturbation size:                 max {moved.max():.3e}  median {moved.median():.3e} rad")
print(f"float32 rung   vs raygen:          max {(direction - reference['bearings']).norm(dim=-1)[valid].max():.3e}")
print(f"float64 exact  vs raygen:          max {(exact.float() - reference['bearings']).norm(dim=-1)[valid].max():.3e}")
print(f"float32 rung   vs float64 exact:   max {(direction - exact.float()).norm(dim=-1)[valid].max():.3e}")

# Where is the disagreement? By theta, and by how far the raygen's own reprojection lands.
chord = (direction - reference["bearings"]).norm(dim=-1)
theta_base = torch.atan2(
    base["bearings"][..., :2].norm(dim=-1), base["bearings"][..., 2]
)
worst = torch.argmax(torch.where(valid, chord, torch.zeros_like(chord)))
row, column = int(worst // WIDTH), int(worst % WIDTH)
print(f"worst pixel ({row},{column}): theta {float(theta_base[row, column]):.4f} rad, "
      f"chord {float(chord[row, column]):.3e}")
for low, high in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.3), (1.3, 1.6)):
    selection = valid & (theta_base >= low) & (theta_base < high)
    if selection.any():
        print(f"  theta in [{low}, {high}): n={int(selection.sum()):5d} "
              f"max chord {chord[selection].max():.3e}")
