#!/usr/bin/env python
"""The seed lot: is `rttpf_z - rttpf` stable, or is n=1 hiding SGD noise?

WHY. Every headline number in this branch is a single run per cell. The load-bearing one is
the decomposition contrast `rttpf_z - rttpf` = +0.244 dB (p=0.016, 7/7) -- re-calibration
plus one non-central degree of freedom against re-calibration alone. If the seed-to-seed
spread of that difference is comparable to +0.244, the seven-scene Wilcoxon is measuring
optimiser noise that happens to be correlated across scenes, and the claim is not safe.

`config.py:117` wires `seed` through `set_seeds` at the very top of `train.py`, before the
scene, the point cloud and the raytracer are built. Note what it does NOT control: the CUDA
atomics in the backward pass are non-deterministic, so two runs at the SAME seed already
differ by ~0.05-0.06 dB. That is the floor this lot measures against, not zero.

WHY SEED 0 IS RE-RUN HERE. The existing seed-0 rttpf runs live in the `rttpf-intrinsics`
worktree, which has since been merged into this one. Mixing them with seeds 1-2 produced
here would confound "seed" with "code state". Re-running seed 0 in this tree costs 14 runs
and removes the confound entirely -- and doubles as a replication check against the other
worktree's numbers, which is worth having on its own.

DESIGN. 7 scenes x {rttpf, rttpf_z} x {0, 1, 2} = 42 runs, ~15.5 min each, strictly serial
(peak VRAM 15.8-18.7 GB does not fit the 12 GB card and two will not fit the 24 GB one).
The contrast is PAIRED WITHIN SEED: both rungs at seed k share the initialisation, so
`rttpf_z(k) - rttpf(k)` isolates the rung. Report the mean over seeds per scene, and the
spread of the per-seed contrast as the honest error bar.

usage:
    python scripts/analysis/seed_lot.py                 # build the queue file
    python scripts/analysis/seed_lot.py --seeds 1 2     # only the new seeds
"""

import argparse
import os

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
RUNGS = ["rttpf", "rttpf_z"]
OUT = "/workspace/gray/worktrees/noncentral-camera/tmp/seeds"
ROOT = "/workspace/gray/worktrees/noncentral-camera"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    args = parser.parse_args()

    env = ("PATH=/workspace/gray/.venv/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID "
           "CUDA_VISIBLE_DEVICES=1")
    # * the guard exists because an unguarded run already OOMed when another session took the
    # * card between the check and the allocation; the retry is because the guard can lose
    # * that race again.
    wait = ('for i in $(seq 1 90); do f=$(nvidia-smi --query-gpu=memory.free '
            '--format=csv,noheader,nounits -i 1); [ "$f" -gt 20000 ] && break; sleep 60; done')
    lines = []
    # * seed-major so that a partial lot is still a COMPLETE lower-n experiment: if the queue
    # * is cut short, every scene has the same number of seeds rather than three scenes
    # * having three seeds and four having none.
    for seed in args.seeds:
        for scene in args.scenes:
            for rung in RUNGS:
                path = f"tmp/seeds/{scene}_{rung}_s{seed}"
                lines.append(
                    f"pueue add --group gpu1 --priority 100 --print-task-id -- "
                    f"'cd {ROOT} && export {env} && for try in 1 2 3; do {wait}; "
                    f"bash scripts/train_myscenes.sh {scene} {path} -y "
                    f"--camera_opt {rung} --camera_opt_from_iter 3000 --seed {seed} "
                    f"&& break; echo retry; sleep 120; done'")
    os.makedirs(OUT, exist_ok=True)
    target = os.path.join(OUT, "queue.sh")
    with open(target, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(lines)} runs, ~{len(lines) * 15.5 / 60:.1f} h serial -> {target}")


if __name__ == "__main__":
    main()
