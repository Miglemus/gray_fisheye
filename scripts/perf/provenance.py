"""Provenance schema for every speed measurement in this project.

WHY THIS EXISTS
---------------
Not one FPS number currently on disk is citable, and the reasons are all provenance
reasons, not measurement reasons:

* four `gray`-rttpf values were written inside a 31 s window, which is impossible
  sequentially (each run rebuilds a BVH over ~5e5 instances and warms up 57 views);
* the same scene, the same resolution and the same model size read 118.87 FPS on one
  track and 149.74 on another;
* five FullCircle values are called contention artefacts by IMPLEMENTATION.md itself;
* SPaGS reports a best-of-10, gray / DFGS / 3dgrut report a single pass, 3DGEER reports
  a Python wall clock scraped from a log;
* **no CSV on disk records which card produced which row.**

A number without provenance cannot be defended, and none of the above is detectable
after the fact.  So the rule this module enforces is: *a measurement carries its own
conditions or it does not exist*.  `collect_fps.py` refuses, loudly, to put a row in a
table when any required field below is missing.

Pure standard library on purpose: DirectFisheye-GS, 3dgrut, SPaGS and 3DGEER each have
their own virtualenv, and `stamp_provenance.py` has to run inside all of them.

SCHEMA (gray.perf.provenance/1)
-------------------------------
Every required field is required because its absence has already produced a wrong
number in this project at least once.  See `REQUIRED` below.

TIERS
-----
``tier = "measured"``  the harness that produced the number also wrote this file, in the
                       same process, and observed the card before *and* after the timed
                       region.  This is the only tier that may be published.
``tier = "attested"``  the sidecar was stamped next to a value produced by some other
                       program (`stamp_provenance.py`).  The GPU identity is real, but
                       the exclusivity was observed *after* the fact, so it is evidence,
                       not proof.  `collect_fps.py` rejects it unless asked.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "gray.perf.provenance/1"
SIDECAR_NAME = "perf_provenance.json"

#: Dotted paths that must be present AND non-null for a row to be citable.
#: Each entry ends with the failure it prevents.
REQUIRED: List[str] = [
    "schema",                                  # so a future schema break is loud
    "tier",                                    # measured vs attested
    "harness",                                 # which program timed it
    "timestamp_utc",                           # the 31-second window is only visible here
    "host",
    "gpu.name",                                # 2080 Ti vs TITAN RTX: a 1.6x factor
    "gpu.uuid",                                # the only unambiguous card identity
    "gpu.pci_bus_id",
    "gpu.driver_version",
    "gpu.memory_total_mib",
    "gpu.memory_free_mib_at_start",            # a nearly-full card means contention
    "gpu.compute_capability",                  # Turing sm_75 RT cores are 1st gen
    "exclusivity.exclusive",                   # the contention artefacts
    "exclusivity.foreign_processes_at_start",
    "env.cuda_visible_devices",
    "env.cuda_device_order",                   # without PCI_BUS_ID torch order is flipped
    "method",
    "run_path",
    "n_gaussians",                             # FPS is meaningless without it
    "resolution.width",
    "resolution.height",
    "views.n",
    "views.context",
    "timing.repeats",                          # SPaGS best-of-10 vs everyone's single pass
    "timing.aggregation",
    "timing.warmup_passes",
    "timing.sync",
    "timing.timed_region",                     # what is actually inside the clock
    "timing.timed_region_excludes",
    "value.fps",
]

#: Fields that are required only for tier="measured" (a post-hoc stamp cannot have them).
REQUIRED_MEASURED: List[str] = [
    "gpu.memory_free_mib_at_end",
    "exclusivity.foreign_processes_at_end",
    "timing.per_repeat_fps",
    "timing.spread_pct",
]


# --------------------------------------------------------------------------------------
# nvidia-smi probes (no pynvml dependency: it is absent from three of the five venvs)
# --------------------------------------------------------------------------------------

def _smi(args: List[str]) -> List[List[str]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not on PATH; cannot establish provenance")
    out = subprocess.run([exe] + args, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    rows = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split(",")])
    return rows


def gpu_info(index: int) -> Dict[str, Any]:
    """Identity + free memory of one physical GPU, indexed as `nvidia-smi` indexes it.

    NOTE the index is the *nvidia-smi* index (PCI bus order), never torch's.  Without
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` the two disagree on this machine and
    ``CUDA_VISIBLE_DEVICES=1`` lands on the small card -- which is exactly how a
    "TITAN RTX" number gets measured on a 2080 Ti.
    """
    fields = ["index", "name", "uuid", "pci.bus_id", "driver_version",
              "memory.total", "memory.free", "compute_cap"]
    rows = _smi([f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
                 "-i", str(index)])
    if not rows:
        raise RuntimeError(f"no GPU at nvidia-smi index {index}")
    r = rows[0]
    return {
        "smi_index": int(r[0]),
        "name": r[1],
        "uuid": r[2],
        "pci_bus_id": r[3],
        "driver_version": r[4],
        "memory_total_mib": int(float(r[5])),
        "memory_free_mib": int(float(r[6])),
        "compute_capability": r[7],
    }


def compute_apps(index: int, exclude_pid: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every CUDA context currently open on that card, minus our own pid.

    This is the check `pueue` cannot do for us: a pueue group serialises its *own*
    tasks, and four runs of this project have already died on OOM (and an unknown
    number produced contention FPS) because a process started outside the group held
    12 GB of the same card.
    """
    try:
        rows = _smi(["--query-compute-apps=pid,used_memory,process_name",
                     "--format=csv,noheader,nounits", "-i", str(index)])
    except RuntimeError:
        return []
    apps = []
    for r in rows:
        if len(r) < 3 or not r[0].isdigit():
            continue
        pid = int(r[0])
        if exclude_pid is not None and pid == exclude_pid:
            continue
        apps.append({"pid": pid, "used_mib": int(float(r[1])), "name": r[2]})
    return apps


def git_info(repo: str) -> Dict[str, Any]:
    def run(*args):
        try:
            o = subprocess.run(["git", "-C", repo] + list(args),
                               capture_output=True, text=True, timeout=30)
            return o.stdout.strip() if o.returncode == 0 else None
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "repo": os.path.abspath(repo),
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status.strip()),
    }


# --------------------------------------------------------------------------------------
# building / validating
# --------------------------------------------------------------------------------------

def env_block() -> Dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "<unset>"),
        "gray_no_ray_cache": os.environ.get("GRAY_NO_RAY_CACHE", "<unset>"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def build(
    *,
    tier: str,
    harness: str,
    method: str,
    run_path: str,
    gpu_index: int,
    n_gaussians: Optional[int],
    width: int,
    height: int,
    n_views: int,
    context: str,
    fps: float,
    timing: Dict[str, Any],
    gpu_start: Dict[str, Any],
    gpu_end: Optional[Dict[str, Any]] = None,
    foreign_start: Optional[List[Dict[str, Any]]] = None,
    foreign_end: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a sidecar document.  Callers pass what they observed; nothing is guessed."""
    if tier not in ("measured", "attested"):
        raise ValueError(f"tier must be 'measured' or 'attested', got {tier!r}")
    foreign_start = list(foreign_start or [])
    doc: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "tier": tier,
        "harness": harness,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_unix": time.time(),
        "host": socket.gethostname(),
        "gpu": {
            "name": gpu_start["name"],
            "uuid": gpu_start["uuid"],
            "pci_bus_id": gpu_start["pci_bus_id"],
            "smi_index": gpu_start["smi_index"],
            "driver_version": gpu_start["driver_version"],
            "compute_capability": gpu_start["compute_capability"],
            "memory_total_mib": gpu_start["memory_total_mib"],
            "memory_free_mib_at_start": gpu_start["memory_free_mib"],
            "memory_free_mib_at_end": None if gpu_end is None else gpu_end["memory_free_mib"],
        },
        "exclusivity": {
            "probe": "nvidia-smi --query-compute-apps",
            "exclusive": len(foreign_start) == 0 and not foreign_end,
            "foreign_processes_at_start": foreign_start,
            "foreign_processes_at_end": None if foreign_end is None else list(foreign_end),
        },
        "env": env_block(),
        "method": method,
        "run_path": os.path.abspath(run_path),
        "n_gaussians": n_gaussians,
        "resolution": {"width": int(width), "height": int(height)},
        "views": {"n": int(n_views), "context": context},
        "timing": dict(timing),
        "value": {"fps": float(fps), "unit": "frames/s"},
    }
    doc["gpu_index_requested"] = gpu_index
    if extra:
        doc["extra"] = extra
    return doc


def _get(doc: Dict[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate(doc: Dict[str, Any], *, strict_tier: str = "measured") -> List[str]:
    """Return the list of reasons this row is NOT citable.  Empty list = citable.

    `strict_tier="measured"` (the default) is the publication gate.  Pass
    `strict_tier="attested"` to accept post-hoc stamps as well.
    """
    problems: List[str] = []
    schema = doc.get("schema")
    if schema != SCHEMA_VERSION:
        problems.append(f"schema is {schema!r}, expected {SCHEMA_VERSION!r}")

    for path in REQUIRED:
        v = _get(doc, path)
        if v is None or v == "" or v == []:
            # an empty foreign-process list is the GOOD case, not a missing field
            if path == "exclusivity.foreign_processes_at_start" and v == []:
                continue
            problems.append(f"missing required field: {path}")

    tier = doc.get("tier")
    if strict_tier == "measured" and tier != "measured":
        problems.append(
            f"tier={tier!r}: stamped after the fact, exclusivity was not observed around "
            "the timed region (re-measure with scripts/perf/bench_fps.py)"
        )
    if tier == "measured":
        for path in REQUIRED_MEASURED:
            if _get(doc, path) is None:
                problems.append(f"missing required field (measured tier): {path}")

    if _get(doc, "exclusivity.exclusive") is False:
        foreign = (_get(doc, "exclusivity.foreign_processes_at_start") or []) + \
                  (_get(doc, "exclusivity.foreign_processes_at_end") or [])
        detail = ", ".join(f"pid {p.get('pid')} ({p.get('used_mib')} MiB)" for p in foreign)
        problems.append("card was NOT exclusive"
                        + (f": {detail}" if detail else " (no process list recorded)"))

    order = _get(doc, "env.cuda_device_order")
    if order != "PCI_BUS_ID":
        problems.append(
            f"CUDA_DEVICE_ORDER={order!r}: torch device order is not nvidia-smi's, so the "
            "recorded card may not be the card that ran")

    spread = _get(doc, "timing.spread_pct")
    if spread is not None and spread > 5.0:
        problems.append(f"repeat spread {spread:.1f} % > 5 %: the card was not in a steady state")

    n = _get(doc, "n_gaussians")
    if n is not None and n <= 0:
        problems.append(f"n_gaussians = {n}")

    return problems


def sidecar_path(value_path: str | os.PathLike) -> Path:
    """Where the sidecar for a run directory or a value file lives."""
    p = Path(value_path)
    return (p / SIDECAR_NAME) if p.is_dir() else (p.parent / SIDECAR_NAME)


def write(doc: Dict[str, Any], path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    return path


def read(path: str | os.PathLike) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
