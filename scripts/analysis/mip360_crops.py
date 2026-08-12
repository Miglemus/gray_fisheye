"""Cut the crops the report shows: where does the learned camera visibly change the render?

Picks, per scene, the test view and the window with the largest drop in squared error
between `off` and `noncentral`, then writes gt / off / noncentral crops side by side at 3x.
The point is to show that the difference is a *sharpness* difference -- the baseline is
fitting a scene through a wrong camera and pays for it in blur -- not a colour or
exposure difference.

Outputs PNGs into tmp/mipnerf360/analysis/crops/ plus an index json.
"""

import json
import os

import numpy as np
from PIL import Image

ROOT = "/workspace/gray/worktrees/noncentral-camera"
OUTDIR = f"{ROOT}/tmp/mipnerf360/analysis/crops"
SCENES = ["bicycle", "bonsai", "garden", "room", "kitchen"]
WIN = 128
ZOOM = 3


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    index = {}
    for scene in SCENES:
        base_off = f"{ROOT}/tmp/mipnerf360/{scene}_off/test/15000/pinhole"
        base_nc = f"{ROOT}/tmp/mipnerf360/{scene}_noncentral/test/15000/pinhole"
        if not os.path.exists(base_nc):
            continue
        names = sorted(os.listdir(f"{base_off}/renders"))
        best = None
        for name in names:
            off = np.asarray(Image.open(f"{base_off}/renders/{name}"), np.float64) / 255
            nc = np.asarray(Image.open(f"{base_nc}/renders/{name}"), np.float64) / 255
            gt = np.asarray(Image.open(f"{base_off}/gt/{name}"), np.float64) / 255
            gain = ((off - gt) ** 2).mean(-1) - ((nc - gt) ** 2).mean(-1)
            # * Box-sum over WIN x WIN via a cumulative sum, then take the best window.
            cumulative = np.pad(gain, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            box = (cumulative[WIN:, WIN:] - cumulative[:-WIN, WIN:]
                   - cumulative[WIN:, :-WIN] + cumulative[:-WIN, :-WIN])
            y, x = np.unravel_index(int(box.argmax()), box.shape)
            score = float(box[y, x])
            if best is None or score > best[0]:
                best = (score, name, x, y)

        score, name, x, y = best
        for tag, base in (("gt", f"{base_off}/gt"), ("off", f"{base_off}/renders"),
                          ("noncentral", f"{base_nc}/renders")):
            crop = Image.open(f"{base}/{name}").crop((x, y, x + WIN, y + WIN))
            crop = crop.resize((WIN * ZOOM, WIN * ZOOM), Image.NEAREST)
            crop.save(f"{OUTDIR}/{scene}_{tag}.png")
        index[scene] = {"view": name, "x": int(x), "y": int(y), "win": WIN, "score": score}
        print(f"{scene:9s} view={name} at ({x},{y}) box_gain={score:.4f}")

    with open(f"{OUTDIR}/index.json", "w") as handle:
        json.dump(index, handle, indent=1)


if __name__ == "__main__":
    main()
