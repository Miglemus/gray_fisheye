"""Collect speed measurements into a table -- and REFUSE, loudly, anything unprovenanced.

    # the table (only citable rows survive)
    python scripts/perf/collect_fps.py --roots tmp/perf

    # why every number currently on disk is not citable
    python scripts/perf/collect_fps.py --legacy-audit \
        /workspace/gray/tmp/final /workspace/gray/worktrees/noncentral-camera/out \
        /workspace/gray/worktrees/fullcircle-erp/out

This is the piece that makes an unciteable number *impossible to publish by accident*.
The rules, and the failure each one blocks:

  no sidecar                -> REJECT.  Every `fps.csv` and `fps.json` on disk today is
                               in this class: none records which card produced it.
  tier = "attested"         -> REJECT unless --allow-attested.  A sidecar stamped after
                               the fact proves the card's identity, not its exclusivity.
  card not exclusive        -> REJECT.  This is the five FullCircle contention artefacts.
  CUDA_DEVICE_ORDER != PCI_BUS_ID -> REJECT.  Without it torch's index is not
                               nvidia-smi's and the sidecar names the wrong card.
  repeat spread > 5 %       -> REJECT.  The card was not in a steady state.
  mixed GPU uuid in a table -> REJECT the whole table.  Comparing a 2080 Ti row against a
                               TITAN RTX row measures the cards, not the methods.
  mixed aggregation         -> REJECT the whole table.  A best-of-10 (SPaGS) and a single
                               pass (gray, DFGS, 3dgrut) are different estimators.
  mixed resolution / views  -> flagged; FPS is a throughput number and both change it.

Exit status is non-zero as soon as one requested row is rejected, so a Makefile or a
figure script cannot silently drop it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import provenance  # noqa: E402

LEGACY_VALUE_FILES = ("fps.csv", "fps.json", "fps_by_scale.json")


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------

def find_sidecars(roots: List[str]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob(provenance.SIDECAR_NAME)))
            out.extend(sorted(p.rglob("*.perf.json")))
    seen, uniq = set(), []
    for f in out:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    return uniq


def find_legacy(roots: List[str]) -> List[Tuple[Path, Optional[float], bool]]:
    """(value file, value, has_sidecar) for every legacy FPS artefact under `roots`."""
    rows = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for name in LEGACY_VALUE_FILES:
            for f in sorted(base.rglob(name)):
                rows.append((f, _read_legacy_value(f),
                             provenance.sidecar_path(f).exists()))
    return rows


def _read_legacy_value(path: Path) -> Optional[float]:
    try:
        txt = path.read_text().strip()
        if path.suffix == ".csv":
            return float(txt.splitlines()[0])
        data = json.loads(txt)
        if isinstance(data, dict):
            for key in ("fps", "1.0", "1"):
                if key in data:
                    return float(data[key])
            vals = [v for v in data.values() if isinstance(v, (int, float))]
            return float(vals[0]) if vals else None
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------------------
# table-level (cross-row) checks
# --------------------------------------------------------------------------------------

def table_checks(docs: List[Dict[str, Any]]) -> List[str]:
    """Failures that only exist between rows.  A single row can be perfect and the table wrong."""
    problems: List[str] = []
    if len(docs) < 2:
        return problems

    def spread(key_fn, label, fmt=str):
        vals = defaultdict(list)
        for d in docs:
            vals[key_fn(d)].append(d.get("run_path", "?"))
        if len(vals) > 1:
            detail = "; ".join(f"{fmt(k)} <- {len(v)} row(s)" for k, v in vals.items())
            problems.append(f"MIXED {label} across the table: {detail}")

    spread(lambda d: d["gpu"]["uuid"], "GPU",
           fmt=lambda u: next((d["gpu"]["name"] + " " + u[:12]
                               for d in docs if d["gpu"]["uuid"] == u), u))
    spread(lambda d: d["timing"]["aggregation"], "aggregation")
    spread(lambda d: d["timing"]["repeats"], "repeat count")
    spread(lambda d: d["timing"]["timed_region"], "timed region")
    spread(lambda d: (d["resolution"]["width"], d["resolution"]["height"]), "resolution")
    spread(lambda d: d["views"]["n"], "view count")
    spread(lambda d: d["gpu"]["driver_version"], "driver version")

    problems.extend(timeline_check(docs))
    return problems


#: A full sequential measurement is scene load + safetensors load + acceleration-structure
#: build + at least one warmup pass + `repeats` timed passes. On the scenes in this project
#: (5e5 instances, 57-106 views) that is tens of seconds, never a handful. This is the
#: check that catches the four `gray`-rttpf values written inside a 31 s window: at 4 rows
#: that is 10 s between consecutive measurements, which no run here can do.
#: It is a HEURISTIC, and it is a *prompt to verify*, not a proof -- hence its own message.
MIN_SECONDS_BETWEEN_MEASUREMENTS = 20.0


def timeline_check(docs: List[Dict[str, Any]],
                   min_gap: float = MIN_SECONDS_BETWEEN_MEASUREMENTS) -> List[str]:
    stamps = sorted(d.get("timestamp_unix") or 0.0 for d in docs)
    if len(stamps) < 3 or stamps[0] <= 0:
        return []
    window = stamps[-1] - stamps[0]
    gaps = len(stamps) - 1
    if window < min_gap * gaps:
        return [(f"IMPLAUSIBLE TIMELINE: {len(stamps)} measurements span {window:.0f} s, "
                 f"i.e. {window / gaps:.1f} s between consecutive rows. A full measurement "
                 f"is scene load + BVH build + warmup + {docs[0]['timing']['repeats']} timed "
                 f"passes and cannot finish that fast. Either these rows did not run "
                 f"sequentially, or they share a process -- check before publishing.")]
    return []


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def report(docs_with_paths: List[Tuple[Path, Dict[str, Any]]], allow_attested: bool,
           show_rejected: bool = True) -> int:
    tier = "attested" if allow_attested else "measured"
    ok: List[Tuple[Path, Dict[str, Any]]] = []
    bad: List[Tuple[Path, Dict[str, Any], List[str]]] = []
    for path, doc in docs_with_paths:
        problems = provenance.validate(doc, strict_tier=tier)
        (ok.append((path, doc)) if not problems else bad.append((path, doc, problems)))

    cross = table_checks([d for _, d in ok])

    print(f"# perf table  ({len(ok)} citable / {len(docs_with_paths)} found)\n")
    if ok:
        hdr = (f"| {'method':<16} | {'run':<44} | {'FPS':>8} | {'spread':>7} | "
               f"{'gauss':>9} | {'res':>11} | {'views':>5} | {'card':<16} | {'when (UTC)':<20} |")
        print(hdr)
        print("|" + "|".join("-" * (len(c) + 2) for c in
                             ["method".ljust(16), "run".ljust(44), "FPS".rjust(8),
                              "spread".rjust(7), "gauss".rjust(9), "res".rjust(11),
                              "views".rjust(5), "card".ljust(16), "when (UTC)".ljust(20)]) + "|")
        for _, d in sorted(ok, key=lambda x: (x[1]["method"], x[1]["run_path"])):
            t = d["timing"]
            run = d["run_path"]
            run = run if len(run) <= 44 else "..." + run[-41:]
            sp = t.get("spread_pct")
            print(f"| {d['method']:<16} | {run:<44} | {d['value']['fps']:8.2f} | "
                  f"{('%.2f%%' % sp) if sp is not None else 'n/a':>7} | "
                  f"{(d['n_gaussians'] or 0):9d} | "
                  f"{d['resolution']['width']}x{d['resolution']['height']:<6} | "
                  f"{d['views']['n']:5d} | {d['gpu']['name'][:16]:<16} | "
                  f"{d['timestamp_utc']:<20} |")
    else:
        print("(nothing citable)")

    if cross:
        print("\n## TABLE REJECTED -- cross-row failures\n")
        for c in cross:
            print(f"  !! {c}")

    if bad and show_rejected:
        print(f"\n## REJECTED rows ({len(bad)})\n")
        for path, d, problems in bad:
            print(f"  {d.get('run_path', path)}")
            for p in problems:
                print(f"      - {p}")

    status = 1 if (bad or cross) else 0
    print(f"\nstatus: {'FAIL' if status else 'OK'}"
          f" ({len(ok)} citable, {len(bad)} rejected, {len(cross)} table-level failures)")
    return status


def legacy_audit(roots: List[str]) -> int:
    rows = find_legacy(roots)
    print(f"# legacy FPS artefacts audit ({len(rows)} value files)\n")
    print("Every row below is UNCITABLE unless it carries a "
          f"{provenance.SIDECAR_NAME} sidecar.\n")
    print(f"| {'value file':<86} | {'FPS':>9} | {'sidecar':<8} | verdict |")
    print("|" + "-" * 88 + "|" + "-" * 11 + "|" + "-" * 10 + "|---------|")
    unciteable = 0
    for path, value, has in sorted(rows):
        p = str(path)
        p = p if len(p) <= 86 else "..." + p[-83:]
        verdict = "citable?" if has else "REJECT: no provenance"
        if not has:
            unciteable += 1
        print(f"| {p:<86} | {('%.2f' % value) if value is not None else 'n/a':>9} | "
              f"{'yes' if has else 'NO':<8} | {verdict} |")
    print(f"\n{unciteable} of {len(rows)} value files have no provenance and cannot be "
          "published. Re-measure with scripts/perf/bench_fps.py.")
    return 1 if unciteable else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=[],
                    help="directories (or sidecar files) to collect")
    ap.add_argument("--legacy-audit", nargs="*", default=None,
                    help="scan these roots for fps.csv/fps.json and report their provenance")
    ap.add_argument("--allow-attested", action="store_true",
                    help="accept post-hoc stamps (evidence, not proof) -- never for a paper")
    ap.add_argument("--json-out", default=None, help="write the citable rows as JSON")
    args = ap.parse_args(argv)

    status = 0
    if args.legacy_audit is not None:
        status |= legacy_audit(args.legacy_audit or ["."])
        if args.roots:
            print()

    if args.roots:
        docs = []
        for f in find_sidecars(args.roots):
            try:
                docs.append((f, provenance.read(f)))
            except Exception as exc:
                print(f"  !! unreadable sidecar {f}: {exc}", file=sys.stderr)
                status = 1
        status |= report(docs, allow_attested=args.allow_attested)
        if args.json_out:
            citable = [d for _, d in docs
                       if not provenance.validate(
                           d, strict_tier="attested" if args.allow_attested else "measured")]
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(citable, indent=2))
            print(f"wrote {args.json_out} ({len(citable)} rows)")

    if not args.roots and args.legacy_audit is None:
        ap.error("pass --roots and/or --legacy-audit")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
