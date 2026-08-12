"""The printable protocol table: what each method's FPS number actually measures.

    python scripts/perf/protocol.py            # the table
    python scripts/perf/protocol.py --diff     # only the columns that make rows incomparable

Every entry below was read out of the source file named in `source`, not inferred, on
2026-08-12.  The point of the table is that **five methods use four different
estimators**, so no cross-method speed row in this project is currently comparable --
and the table says exactly which column breaks each comparison.

If you change a harness, change its row here in the same commit.  `test_perf.py`
asserts the new harness's row matches what `bench_fps.py` actually writes into its
sidecars, so this file cannot drift from the code silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
# `bench_fps` imports torch lazily (inside functions), so this stays CPU-safe. Taking the
# strings from the harness itself is what makes the table unable to drift from the code.
from bench_fps import TIMED_REGION as _NEW_TIMED_REGION  # noqa: E402
from bench_fps import TIMED_REGION_EXCLUDES as _NEW_EXCLUDES  # noqa: E402

COLUMNS = ["repetitions", "aggregation", "warmup", "synchronisation",
           "timed region", "readback", "provenance"]

#: method -> column -> value.  `source` is where it was read.
PROTOCOLS: Dict[str, Dict[str, str]] = {
    "gray (measure_fps.py, current)": {
        "repetitions": "1",
        "aggregation": "single pass",
        "warmup": "1 pass over all views (skip_copy=True)",
        "synchronisation": "cuda events + synchronize() around the whole loop",
        "timed region": "loop over test views, raytracer(cam, skip_copy=True), no_grad",
        "readback": "none (skip_copy skips even the device-side clone)",
        "provenance": "NONE - writes a bare float into fps.csv",
        "source": "measure_fps.py",
    },
    "DirectFisheye-GS": {
        "repetitions": "1",
        "aggregation": "single pass",
        "warmup": "1 untimed pass over all views",
        "synchronisation": "cuda events + synchronize() around the whole loop",
        "timed region": "loop over test views, render(), no image saving",
        "readback": "none",
        "provenance": "NONE",
        "source": "/workspace/DirectFisheye-GS/worktrees/rttpf/measure_fps.py",
    },
    "3dgrut / 3DGUT": {
        "repetitions": "1",
        "aggregation": "single pass",
        "warmup": "1 untimed pass; batches pre-fetched to GPU first",
        "synchronisation": "cuda events + synchronize() around the whole loop",
        "timed region": "loop over pre-fetched gpu batches, model(batch)",
        "readback": "none",
        "provenance": "NONE - fps.json = {'fps': float}",
        "source": "/workspace/3dgrut/worktrees/rttpf/threedgrut/render.py:250",
    },
    "SPaGS": {
        "repetitions": "10",
        "aggregation": "BEST of 10  <-- different estimator",
        "warmup": "2 passes over all views",
        "synchronisation": "cuda events + synchronize() per repeat",
        "timed region": "loop over test views, render()",
        "readback": "none",
        "provenance": "NONE",
        "source": "/workspace/SPaGS-rttpf/nerficg/src/Methods/SPaGS/measure_fps.py",
    },
    "3DGEER": {
        "repetitions": "1",
        "aggregation": "mean of per-view wall times",
        "warmup": "unknown / not explicit",
        "synchronisation": "NONE - Python wall clock  <-- measures launch queueing",
        "timed region": "render() call inside the repo's own render.py, scraped from its log",
        "readback": "unknown",
        "provenance": "NONE - regex-scraped out of render_kb.log",
        "source": "/workspace/3dgeer/scripts_myscenes/collect_sidecars.py:33",
    },
    "gray (scripts/perf/bench_fps.py, NEW)": {
        "repetitions": "3 (--repeats)",
        "aggregation": "median",
        "warmup": "1 pass over all views (--warmup-passes)",
        "synchronisation": "synchronize() before start.record() and after end.record(), every repeat",
        # taken from the harness, not retyped: see the import at the top of this file
        "timed region": _NEW_TIMED_REGION,
        "excludes": "; ".join(_NEW_EXCLUDES),
        "readback": "none (skip_copy=True)",
        "provenance": "perf_provenance.json on EVERY measurement; exclusivity probed "
                      "before and after; collect_fps.py rejects a row without it",
        "source": "scripts/perf/bench_fps.py",
    },
}

#: Why each disagreement matters, keyed by column.
CONSEQUENCE: Dict[str, str] = {
    "repetitions": "different sample sizes -> different variance, and best-of needs many",
    "aggregation": "a best-of is the FLOOR of the contention distribution; a single pass is "
                   "a draw from it. On a shared card the gap is the contention itself.",
    "warmup": "an un-warmed pass pays OptiX pipeline compile / BVH upload / cache fill once, "
              "which on a 57-view pass is a several-percent tax on the first method only",
    "synchronisation": "a Python wall clock without synchronize() times kernel LAUNCHES, "
                       "not kernel execution; it can be arbitrarily optimistic",
    "timed region": "if one method's clock contains the readback and another's does not, "
                    "the difference is a memcpy, not a renderer",
    "readback": "same, isolated: device->host is ~O(W*H*3*4) bytes per frame",
    "provenance": "without the card identity a 2080 Ti row and a TITAN RTX row look alike; "
                  "on this machine that alone is a ~1.6x factor",
}


def render_table(fmt: str = "md") -> str:
    names = list(PROTOCOLS)
    lines: List[str] = []
    if fmt == "md":
        lines.append("| " + " | ".join(["method"] + COLUMNS) + " |")
        lines.append("|" + "|".join(["---"] * (len(COLUMNS) + 1)) + "|")
        for n in names:
            lines.append("| " + " | ".join([n] + [PROTOCOLS[n][c] for c in COLUMNS]) + " |")
        lines.append("")
        lines.append("| method | source (read, not assumed) |")
        lines.append("|---|---|")
        for n in names:
            lines.append(f"| {n} | `{PROTOCOLS[n]['source']}` |")
    else:
        for n in names:
            lines.append(n)
            for c in COLUMNS:
                lines.append(f"    {c:<18} {PROTOCOLS[n][c]}")
            lines.append(f"    {'source':<18} {PROTOCOLS[n]['source']}")
            lines.append("")
    return "\n".join(lines)


def render_diff() -> str:
    lines: List[str] = ["# Columns on which the methods DISAGREE", ""]
    any_diff = False
    for c in COLUMNS:
        values: Dict[str, List[str]] = {}
        for n, p in PROTOCOLS.items():
            values.setdefault(p[c], []).append(n)
        if len(values) > 1:
            any_diff = True
            lines.append(f"## {c}")
            lines.append(f"consequence: {CONSEQUENCE[c]}")
            for v, ns in values.items():
                lines.append(f"  - {v}")
                for n in ns:
                    lines.append(f"      {n}")
            lines.append("")
    if not any_diff:
        lines.append("(none -- every method uses the same protocol)")
    lines.append("VERDICT: a cross-method speed table drawn from these harnesses as they "
                 "stand is NOT valid. Re-measure every method through one harness, on one "
                 "card, back to back, with a provenance sidecar per row.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["md", "text"], default="md")
    ap.add_argument("--diff", action="store_true",
                    help="print only the columns that make the rows incomparable")
    a = ap.parse_args(argv)
    print(render_diff() if a.diff else render_table(a.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
