"""CPU tests for `scripts/traversal_coherence.py` -- the BVH-coherence measurement.

The script answers the one question a graphics reviewer always asks about a non-central
camera: does losing the common ray origin cost traversal coherence? Its `--from-runs` mode
reads `traversal_stats.csv` off runs already on disk, so it is CPU-only and testable here;
its `--measure` mode renders and is not exercised by this file (no GPU may be taken).

What is worth pinning down, and why each of these has a real failure behind it:

* **The CSV format is inconsistent with itself.** `train.py:217` writes a COMMA-separated
  header and `train.py:511` writes SPACE-separated rows. A `split(",")` parser returns one
  field per row and either crashes or -- worse -- silently reads the iteration number as the
  hits count. Same trap in `num_gaussians.csv`.
* **The rung guard used to be a substring test.** `"z" not in rung and rung != "noncentral"`
  rejected `noncentral_no_ana` -- a rung that *does* move the origin, and the one the error
  message it printed recommended. A wrong guard here wastes a queued GPU job, or worse,
  admits a central rung and reports its z-on / z-off arms as a real comparison of nothing.
* **A ratio is only a control if the pairing is right.** The last row of each CSV is the one
  to compare, and a pair with mismatched iteration counts must be flagged, not averaged in.

    python -m pytest tests/test_traversal_coherence.py -q
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "traversal_coherence", REPO_ROOT / "scripts" / "traversal_coherence.py"
)
traversal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traversal)


def write_run(directory: Path, rows, gaussians=None):
    """A run directory as `train.py` writes it: comma header, space-separated rows."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["iteration,num_hit_per_ray,num_accum_per_ray"]
    lines += [f"{iteration} {hits:.2f} {accum:.2f}" for iteration, hits, accum in rows]
    (directory / "traversal_stats.csv").write_text("\n".join(lines) + "\n")
    if gaussians is not None:
        text = "iteration num_gaussians\n" + "\n".join(
            f"{iteration:05d} {count}" for iteration, count in gaussians
        )
        (directory / "num_gaussians.csv").write_text(text + "\n")


# ------------------------------------------------------------------------ CSV parsing


def test_reads_the_space_separated_rows_under_a_comma_separated_header(tmp_path):
    write_run(tmp_path / "run", [(1000, 8.5, 6.5), (2000, 9.03, 7.18)])
    last = traversal.read_traversal_csv(tmp_path / "run")
    assert last == {"iteration": 2000, "hits_per_ray": 9.03, "accum_per_ray": 7.18}


def test_missing_or_empty_csv_is_none_not_a_crash(tmp_path):
    assert traversal.read_traversal_csv(tmp_path / "absent") is None
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "traversal_stats.csv").write_text(
        "iteration,num_hit_per_ray,num_accum_per_ray\n"
    )
    assert traversal.read_traversal_csv(tmp_path / "empty") is None


def test_reads_the_last_gaussian_count(tmp_path):
    write_run(tmp_path / "run", [(1000, 8.0, 6.0)], gaussians=[(1000, 200), (2000, 131059)])
    assert traversal.read_last_gaussians(tmp_path / "run") == 131059
    assert traversal.read_last_gaussians(tmp_path / "absent") is None


# ---------------------------------------------------------------------------- pairing


def test_pairs_report_the_ratio_and_flag_a_length_mismatch(tmp_path):
    write_run(tmp_path / "off", [(2000, 10.0, 8.0)], gaussians=[(2000, 1000)])
    write_run(tmp_path / "nc", [(2000, 11.0, 9.0)], gaussians=[(2000, 990)])
    write_run(tmp_path / "short", [(1000, 11.0, 9.0)], gaussians=[(1000, 990)])

    rows = traversal.from_runs(tmp_path, [("pair", "off", "nc"), ("uneven", "off", "short")])
    matched, uneven = rows
    assert matched["present"] and matched["iterations_match"]
    assert matched["hits_per_ray_ratio"] == pytest.approx(1.1)
    assert matched["accum_per_ray_ratio"] == pytest.approx(1.125)
    assert matched["gaussian_ratio"] == pytest.approx(0.99)
    assert uneven["iterations_match"] is False, "a pair at different iterations must be flagged"


def test_a_missing_run_is_reported_not_skipped(tmp_path):
    write_run(tmp_path / "off", [(2000, 10.0, 8.0)])
    rows = traversal.from_runs(tmp_path, [("gone", "off", "does_not_exist")])
    assert rows[0]["present"] is False
    assert "hits_per_ray_ratio" not in rows[0]
    assert traversal.summarize(rows) == {"n": 0}


# --------------------------------------------------------------------------- summary


def test_summary_is_the_paired_statistic_over_present_pairs():
    results = [
        {"present": True, "hits_per_ray_ratio": r, "accum_per_ray_ratio": r, "gaussian_ratio": 1.0}
        for r in (0.98, 1.00, 1.02, 1.04)
    ] + [{"present": False, "label": "missing"}]
    summary = traversal.summarize(results)
    assert summary["n"] == 4, "an absent pair must not enter the statistic"
    assert summary["hits_ratio_mean"] == pytest.approx(1.01)
    assert summary["hits_ratio_median"] == pytest.approx(1.01)
    assert summary["hits_ratio_min"] == pytest.approx(0.98)
    assert summary["hits_ratio_max"] == pytest.approx(1.04)
    low, high = summary["hits_ratio_ci95"]
    assert low < 1.01 < high


def test_summary_survives_without_the_gaussian_counts():
    results = [
        {"present": True, "hits_per_ray_ratio": 1.0, "accum_per_ray_ratio": 1.0, "gaussian_ratio": None}
        for _ in range(3)
    ]
    summary = traversal.summarize(results)
    assert summary["n"] == 3
    assert "spearman_hits_vs_gaussian_ratio" not in summary


# -------------------------------------------------------------- the GPU mode's arithmetic


def test_shared_mask_pools_only_pixels_both_arms_traced():
    """The bias this removes is real: a per-arm mask changes the denominator per arm.

    Here pixel 3 is traced only by the non-central arm and is expensive (10 hits). Under
    per-arm masks the non-central arm would average (4+10)/2 = 7 against the z0 arm's 4, i.e.
    a 1.75x "coherence cost" produced entirely by one pixel entering the traced set. Pooled
    over the shared mask both arms give 4 and the ratio is exactly 1.
    """
    import torch

    hits_nc = torch.tensor([[0.0, 4.0, 10.0]])
    hits_z0 = torch.tensor([[0.0, 4.0, 0.0]])
    stats = traversal.shared_mask_stats(hits_nc, hits_z0, hits_nc, hits_z0, num_views=1)

    assert stats["shared_traced_pixels"] == 1
    assert stats["shared_traced_pixel_fraction"] == pytest.approx(1.0 / 3.0)
    assert stats["noncentral_hits_per_shared_ray"] == pytest.approx(4.0)
    assert stats["z0_hits_per_shared_ray"] == pytest.approx(4.0)


def test_shared_mask_divides_by_the_number_of_views():
    "The counters accumulate over the whole pass, exactly as train.py's column does."
    import torch

    hits = torch.tensor([[6.0, 12.0]])
    stats = traversal.shared_mask_stats(hits, hits, hits, hits, num_views=3)
    assert stats["noncentral_hits_per_shared_ray"] == pytest.approx(3.0)


def test_shared_mask_refuses_a_degenerate_measurement():
    "Nothing traced by both arms means the render failed; returning 0/0 would hide it."
    import torch

    zeros = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        traversal.shared_mask_stats(zeros, zeros, zeros, zeros, num_views=1)


# ------------------------------------------------------------------------ rung guard


@pytest.mark.parametrize("rung", ["noncentral", "noncentral_no_ana", "z_only"])
def test_rungs_that_move_the_origin_are_accepted(rung):
    assert traversal.rung_is_noncentral(rung) is True


@pytest.mark.parametrize(
    "rung", ["off", "passthrough", "tilt", "radial", "ana", "central_matched", "raxel"]
)
def test_central_rungs_are_refused(rung):
    "A central rung has one origin per view, so a z-on / z-off pair would compare nothing."
    assert traversal.rung_is_noncentral(rung) is False


def test_an_unknown_rung_raises_instead_of_guessing():
    with pytest.raises(KeyError):
        traversal.rung_is_noncentral("noncentral_v2")
