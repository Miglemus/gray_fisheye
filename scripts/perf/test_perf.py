"""CPU-only tests for the perf harness.  No GPU, no render, no pueue task.

    python -m pytest scripts/perf/test_perf.py -q -p no:cacheprovider

They cover the four things that, if they broke silently, would put an unciteable number
back into a table:

1. `provenance.validate` rejects exactly the defects that produced the current mess
   (no card recorded, contention, best-of vs single pass, unstable repeats, post-hoc
   stamps) and accepts a complete document.
2. `collect_fps` catches the cross-row failures a per-row check cannot see: two different
   cards in one table, two different aggregations, and the physically impossible
   31-second window that four `gray`-rttpf values were written in.
3. The FoV -> intrinsics mapping is exact, and the 90 deg wall in gray's fisheye raygen is
   enforced rather than discovered at render time.
4. The claim the whole FoV protocol rests on -- `convert/` round-trips a gray checkpoint
   losslessly -- is re-verified on a real checkpoint, INCLUDING the part of it that is
   false: `vignetting.*` and `camera_model.*` are dropped.

Test 4 skips when no checkpoint is on disk, so the file stays runnable on a bare clone.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "convert"))

import collect_fps  # noqa: E402
import fovmath  # noqa: E402
import protocol  # noqa: E402
import provenance  # noqa: E402


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def _good_doc(**over):
    doc = {
        "schema": provenance.SCHEMA_VERSION,
        "tier": "measured",
        "harness": "scripts/perf/bench_fps.py",
        "timestamp_utc": "2026-08-12T10:00:00Z",
        "timestamp_unix": 1786_600_000.0,
        "host": "box",
        "gpu": {
            "name": "NVIDIA TITAN RTX", "uuid": "GPU-96c55f62", "pci_bus_id": "00000000:65:00.0",
            "smi_index": 1, "driver_version": "570.207", "compute_capability": "7.5",
            "memory_total_mib": 24576, "memory_free_mib_at_start": 24000,
            "memory_free_mib_at_end": 20000,
        },
        "exclusivity": {"probe": "nvidia-smi", "exclusive": True,
                        "foreign_processes_at_start": [], "foreign_processes_at_end": []},
        "env": {"cuda_visible_devices": "1", "cuda_device_order": "PCI_BUS_ID",
                "python": "3.11.0", "platform": "linux", "pid": 1},
        "method": "gray",
        "run_path": "/tmp/run",
        "n_gaussians": 518049,
        "resolution": {"width": 1368, "height": 912},
        "views": {"n": 57, "context": "test"},
        "timing": {
            "repeats": 3, "aggregation": "median", "warmup_passes": 1,
            "sync": "synchronize around every repeat",
            "timed_region": "loop over views, raytracer(cam, skip_copy=True), no_grad",
            "timed_region_excludes": ["scene load", "BVH build", "warmup"],
            "per_repeat_fps": [112.0, 112.5, 113.0], "fps_median": 112.5,
            "spread_pct": 0.89,
        },
        "value": {"fps": 112.5, "unit": "frames/s"},
        "gpu_index_requested": 1,
    }
    for k, v in over.items():
        doc[k] = v
    return doc


def _drop(doc, dotted):
    doc = copy.deepcopy(doc)
    cur = doc
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur.pop(parts[-1], None)
    return doc


# --------------------------------------------------------------------------------------
# 1. per-row validation
# --------------------------------------------------------------------------------------

def test_a_complete_document_is_citable():
    assert provenance.validate(_good_doc()) == []


@pytest.mark.parametrize("field", provenance.REQUIRED)
def test_every_required_field_is_actually_required(field):
    """Removing ANY required field must make the row uncitable.

    This is the non-regression test for the schema itself: adding a field to REQUIRED
    without the validator reading it would be a silent no-op.
    """
    problems = provenance.validate(_drop(_good_doc(), field))
    assert any(field in p for p in problems), (field, problems)


def test_a_row_with_no_card_recorded_is_rejected():
    """The defect shared by 100 % of the fps.csv files on disk today."""
    doc = _drop(_good_doc(), "gpu.uuid")
    doc = _drop(doc, "gpu.name")
    problems = provenance.validate(doc)
    assert any("gpu.uuid" in p for p in problems)
    assert any("gpu.name" in p for p in problems)


def test_contention_is_rejected():
    """The five FullCircle values IMPLEMENTATION.md calls contention artefacts."""
    doc = _good_doc()
    doc["exclusivity"] = {"probe": "nvidia-smi", "exclusive": False,
                          "foreign_processes_at_start": [{"pid": 42, "used_mib": 12000,
                                                          "name": "python"}],
                          "foreign_processes_at_end": []}
    problems = provenance.validate(doc)
    assert any("NOT exclusive" in p for p in problems), problems


def test_unstable_repeats_are_rejected():
    doc = _good_doc()
    doc["timing"] = dict(doc["timing"], per_repeat_fps=[100.0, 130.0, 149.74],
                         fps_median=130.0, spread_pct=38.3)
    assert any("spread" in p for p in provenance.validate(doc))


def test_wrong_device_order_is_rejected():
    """Without CUDA_DEVICE_ORDER=PCI_BUS_ID the sidecar may name the wrong card."""
    doc = _good_doc()
    doc["env"] = dict(doc["env"], cuda_device_order="<unset>")
    assert any("CUDA_DEVICE_ORDER" in p for p in provenance.validate(doc))


def test_attested_tier_is_rejected_by_default_and_accepted_on_request():
    doc = _good_doc(tier="attested")
    doc["timing"] = dict(doc["timing"], per_repeat_fps=None, spread_pct=None)
    doc["gpu"] = dict(doc["gpu"], memory_free_mib_at_end=None)
    doc["exclusivity"] = dict(doc["exclusivity"], foreign_processes_at_end=None)
    assert any("stamped after the fact" in p for p in provenance.validate(doc))
    assert provenance.validate(doc, strict_tier="attested") == []


# --------------------------------------------------------------------------------------
# 2. cross-row (table-level) validation
# --------------------------------------------------------------------------------------

def test_two_different_cards_in_one_table_is_a_table_level_failure():
    a = _good_doc()
    b = _good_doc()
    b["gpu"] = dict(b["gpu"], name="NVIDIA GeForce RTX 2080 Ti", uuid="GPU-4c9d65cb")
    b["timestamp_unix"] = a["timestamp_unix"] + 600
    problems = collect_fps.table_checks([a, b])
    assert any("MIXED GPU" in p for p in problems), problems


def test_best_of_and_single_pass_cannot_share_a_table():
    a = _good_doc()                                       # gray: median of 3
    b = _good_doc()
    b["timing"] = dict(b["timing"], aggregation="best of 10", repeats=10)
    b["timestamp_unix"] = a["timestamp_unix"] + 600
    problems = collect_fps.table_checks([a, b])
    assert any("MIXED aggregation" in p for p in problems), problems


def test_the_31_second_window_is_caught():
    """Four values written inside 31 s cannot have been measured sequentially.

    Each one rebuilds a BVH over ~5e5 instances and warms up 57 views; 7.75 s per full
    measurement is physically impossible. The collector must say so instead of averaging
    them into a table.
    """
    t0 = 1786_600_000.0
    docs = []
    for i, dt in enumerate((0.0, 9.0, 20.0, 31.0)):
        d = _good_doc()
        d["timestamp_unix"] = t0 + dt
        d["run_path"] = f"/tmp/run{i}"
        docs.append(d)
    problems = collect_fps.table_checks(docs)
    assert any("IMPLAUSIBLE TIMELINE" in p for p in problems), problems


def test_a_plausible_timeline_passes():
    t0 = 1786_600_000.0
    docs = []
    for i in range(4):
        d = _good_doc()
        d["timestamp_unix"] = t0 + 120.0 * i
        d["run_path"] = f"/tmp/run{i}"
        docs.append(d)
    assert collect_fps.table_checks(docs) == []


def test_legacy_audit_flags_a_sidecarless_fps_csv(tmp_path):
    run = tmp_path / "somerun"
    run.mkdir()
    (run / "fps.csv").write_text("118.87\n")
    rows = collect_fps.find_legacy([str(tmp_path)])
    assert len(rows) == 1
    path, value, has_sidecar = rows[0]
    assert value == pytest.approx(118.87)
    assert has_sidecar is False


# --------------------------------------------------------------------------------------
# 3. the FoV axis
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("fov", [60.0, 90.0, 120.0, 175.0])
def test_equidistant_intrinsics_put_the_field_edge_exactly_on_the_inscribed_circle(fov):
    w = h = 1024
    fx, fy, cx, cy, *k = fovmath.equidistant_intrinsics(w, h, fov)
    assert k == [0.0, 0.0, 0.0, 0.0]           # exact equidistant, nothing to invert
    assert fx == fy and (cx, cy) == (w / 2, h / 2)
    theta_max = math.radians(fov) / 2.0
    assert fx * theta_max == pytest.approx(min(w, h) / 2.0)   # r(theta_max) = R


@pytest.mark.parametrize("fov", [60.0, 90.0, 120.0])
def test_pinhole_control_matches_the_same_inscribed_field(fov):
    w, h = 1024, 1024
    p = fovmath.pinhole_fov_y(w, h, fov)
    assert p["focal"] * math.tan(math.radians(fov) / 2.0) == pytest.approx(min(w, h) / 2.0)
    assert math.degrees(p["fov_y"]) == pytest.approx(fov)


def test_the_180_degree_wall_is_enforced_not_discovered():
    """`cuda/core/opencv_fisheye.cuh:31` returns an INVALID bearing past theta=90 deg.

    Without this guard the 200 deg point of the sweep would render a black annulus and
    read as a speed-up: fewer hits, more background. That is the class of silent bug this
    project has already paid for twice.
    """
    fovmath.check_gray_can_trace(175.0)
    for fov in (180.0, 190.0, 200.0, 220.0):
        with pytest.raises(ValueError, match="caps theta at 90"):
            fovmath.check_gray_can_trace(fov)
    with pytest.raises(ValueError):
        fovmath.sweep_plan(1024, 1024, [200.0])


def test_sweep_plan_has_a_rectilinear_control_only_where_it_is_defined():
    plan = fovmath.sweep_plan(1024, 1024)
    fish = [p["fov_deg"] for p in plan if p["arm"] == "fisheye_equidistant"]
    pin = [p["fov_deg"] for p in plan if p["arm"] == "pinhole_control"]
    assert fish == fovmath.SWEEP_FOV_DEG
    assert pin == fovmath.PINHOLE_CONTROL_FOV_DEG
    assert max(pin) <= fovmath.PINHOLE_MAX_FOV_DEG


def _fake_cams(n=3):
    import numpy as np
    from gray.camera import CameraInfo
    return [CameraInfo(uid=7, R=np.eye(3), T=np.zeros(3), origin=np.zeros(3),
                       fov_y=1.0, fov_x=1.0, image_path="", image_name=f"{i}.png",
                       image_width=800, image_height=600, is_test=False,
                       model="rad_tan_thin_prism_fisheye",
                       intrinsics=np.arange(16, dtype=np.float64))
            for i in range(n)]


def test_patch_cameras_writes_the_synthetic_lens_and_keeps_the_pose():
    import fov_sweep
    cams = _fake_cams()
    point = fovmath.sweep_plan(1024, 1024, [120.0])[0]
    out = fov_sweep.patch_cameras(cams, point, 1024, 1024, None)
    assert len(out) == 3
    for c, orig in zip(out, cams):
        assert c.model == "opencv_fisheye"
        assert list(c.intrinsics) == point["intrinsics"]
        # image size == render size, so upload_camera_intrinsics' rescale is the identity
        assert (c.image_width, c.image_height) == (1024, 1024)
        assert c.is_test is True
        assert (orig.R == c.R).all() and (orig.origin == c.origin).all()
    # the source cameras must not have been mutated
    assert cams[0].model == "rad_tan_thin_prism_fisheye"


def test_every_sweep_point_gets_a_distinct_uid():
    """Guards the stale-bearing collision class (IMPLEMENTATION.md, 2026-08-07).

    The base-bearing and ray caches key on (uid, model, fov_y, image size, intrinsics), so
    the intrinsics already separate two field angles. Distinct uids make the invariant
    visible instead of implicit -- a 12 dB bug once hid exactly here.
    """
    import fov_sweep
    cams = _fake_cams(2)
    uids = []
    for point in fovmath.sweep_plan(1024, 1024):
        uids += [c.uid for c in fov_sweep.patch_cameras(cams, point, 1024, 1024, None)]
    assert len(uids) == len(set(uids)), "two sweep points share a camera uid"


def test_patch_cameras_honours_the_view_budget():
    import fov_sweep
    point = fovmath.sweep_plan(1024, 1024, [60.0])[0]
    assert len(fov_sweep.patch_cameras(_fake_cams(5), point, 1024, 1024, 2)) == 2
    assert len(fov_sweep.patch_cameras(_fake_cams(5), point, 1024, 1024, None)) == 5


def test_angular_sampling_density_is_reported_because_it_cannot_be_held_fixed():
    lo = fovmath.pixels_per_steradian(1024, 1024, 60.0)
    hi = fovmath.pixels_per_steradian(1024, 1024, 175.0)
    assert lo > hi                      # a wider field samples the sphere more coarsely
    assert lo / hi > 5.0                # ... by a lot: this must be a reported column


# --------------------------------------------------------------------------------------
# 4. the transfer claim the whole protocol rests on
# --------------------------------------------------------------------------------------

CANDIDATE_CHECKPOINTS = [
    _ROOT / "out/fullcircle_rttpf/room1_refit_rttpf_off/gaussians_15000.safetensors",
    Path("/workspace/gray/tmp/final/tunnel_noncentral/gaussians_15000.safetensors"),
    _ROOT / "tmp/ladder_final/off/gaussians_07500.safetensors",
]


def _find_checkpoint():
    for p in CANDIDATE_CHECKPOINTS:
        if p.exists():
            return p
    return None


def test_gray_to_ply_roundtrip_is_bit_exact_on_the_gaussians_and_drops_the_rest():
    """The README says the parameter conversion is lossless. It is -- of the GAUSSIANS.

    It is NOT lossless of the checkpoint: `safetensors_to_ply` writes only the seven
    3DGS-shaped tensors, so `vignetting.*` and `camera_model.lenses.*` are silently
    dropped. For the FoV sweep that is harmless (use an `off` run, and vignetting is a
    per-pixel gain, not a cost), but anyone who transfers a `noncentral` model to another
    engine and compares QUALITY is comparing two different models.
    """
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("no gray checkpoint on disk")
    torch = pytest.importorskip("torch")
    pytest.importorskip("plyfile")
    import safetensors.torch
    from safetensors_ply_conversion import ply_to_safetensors, safetensors_to_ply

    src = safetensors.torch.load_file(str(ckpt))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        safetensors_to_ply(ckpt, d / "p.ply")
        ply_to_safetensors(d / "p.ply", d / "r.safetensors")
        rt = safetensors.torch.load_file(str(d / "r.safetensors"))

    gaussian_keys = {"mean", "opacity", "rotation", "scale",
                     "sh_coeffs_dc", "sh_coeffs_rest", "current_sh_degree"}
    assert gaussian_keys <= set(src), sorted(src)
    for k in sorted(gaussian_keys):
        assert src[k].shape == rt[k].shape, k
        assert torch.equal(src[k], rt[k]), f"{k} changed across the round-trip"

    dropped = set(src) - set(rt)
    assert dropped <= {"vignetting.coefficients", "vignetting.principal_point"} | {
        k for k in src if k.startswith("camera_model.")}, dropped
    if "vignetting.coefficients" in src:
        assert "vignetting.coefficients" in dropped, (
            "vignetting survived the round-trip; update the protocol note in README.md")


# --------------------------------------------------------------------------------------
# the protocol table must not drift from the harness
# --------------------------------------------------------------------------------------

def test_protocol_table_matches_what_the_harness_writes():
    """If `bench_fps.py`'s timed region changes, the printed protocol table must follow."""
    import bench_fps
    row = protocol.PROTOCOLS["gray (scripts/perf/bench_fps.py, NEW)"]
    assert row["timed region"] == bench_fps.TIMED_REGION
    assert row["excludes"] == "; ".join(bench_fps.TIMED_REGION_EXCLUDES)
    assert "median" in row["aggregation"]
    assert "synchronize" in row["synchronisation"]
    for excl in ("BVH", "warmup"):
        assert any(excl in e for e in bench_fps.TIMED_REGION_EXCLUDES)


def test_only_one_raytracer_per_process_is_allowed():
    """A second `Raytracer` aborts at teardown (PipelineWrapper::~PipelineWrapper).

    The FoV sweep would otherwise construct 11 of them. The guard must fail loudly in
    Python rather than let the process abort halfway through a sweep -- and the single
    load is also what keeps the acceleration structure identical across field angles.
    """
    import bench_fps
    saved = bench_fps._LOADED
    try:
        bench_fps._LOADED = object()
        with pytest.raises(SystemExit, match="second Raytracer"):
            bench_fps.GrayBench(run_dir="/nonexistent", smi_index=0, context="test",
                                width=None, height=None, allow_shared=True)
    finally:
        bench_fps._LOADED = saved


def test_protocol_diff_reports_the_estimator_disagreement():
    txt = protocol.render_diff()
    assert "aggregation" in txt
    assert "BEST of 10" in txt
    assert "NOT valid" in txt
