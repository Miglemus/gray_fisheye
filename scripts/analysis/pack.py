"""Assemble every figure's data into one payload the report page embeds verbatim."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]


def read(name):
    path = os.path.join(HERE, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def round_list(values, digits=6):
    return [None if v is None else float(f"%.{digits}g" % v) for v in values]


def main():
    optics, fields, rings, rungs, crops = (read(n) for n in
                                           ("optics.json", "fields.json", "rings.json",
                                            "rungs.json", "crops.json"))
    metrics = json.load(open("/workspace/gray/tmp/final/full_metrics.json"))

    payload = {
        "theta_deg": fields["theta_deg"],
        "bin_centres_deg": optics["bin_centres_deg"],
        "bin_edges_deg": optics["bin_edges_deg"],
        "ring_edges": rings["scenes"]["tunnel"]["ring_edges"],
        "scenes": {},
        "metrics": metrics,
        # * dev ladder: tunnel, -r 8, 7500 iterations, unfreeze at 20%. Kept separate from
        # * the r4/15k numbers -- different resolution and schedule, never to be mixed.
        "dev_ladder": {"off": 27.40, "passthrough": 27.46, "tilt": 27.39, "radial": 27.51,
                       "ana": 27.39, "noncentral": 27.55, "central_matched": 27.39,
                       "raxel": 27.59},
        "crops": crops,
    }

    for scene in SCENES:
        o = optics["scenes"][scene]
        f = fields["scenes"][scene]
        r = rings["scenes"][scene]["methods"]
        baseline_key = "gray-workshopfix" if scene == "workshop" else "gray"
        entry = {
            "focal_px": o["focal_px"],
            "camera_extent": o["camera_extent"],
            "depth": [o["depth_p05"], o["depth_p50"], o["depth_p95"]],
            "n_obs": o["n_observations"],
            "n_views": r["gray"]["views"],
            "profile_z": round_list(o["profile"]["z"]),
            "profile_dtheta": round_list(o["profile"]["dtheta"]),
            "profile_dphi": round_list(o["profile"]["dphi"]),
            "caustic_x": round_list(o["caustic_x"]),
            "caustic_z": round_list(o["caustic_z"]),
            "r_px": f["r_px"],
            "drdtheta_px": f["drdtheta_px"],
            "dtheta_channels": f["noncentral"]["dtheta_channels"],
            "dphi_channels": f["noncentral"]["dphi_channels"],
            "omega": f["noncentral"]["omega"],
            "irreducible_px": round_list(o["irreducible_px"], 4),
            "mean_shift_px": round_list(o["mean_shift_px"], 4),
            "depth_hist": o["depth_hist"],
            "depth_hist_log_edges": round_list(o["depth_hist_log_edges"], 5),
            "psnr": {
                "gray_published": rings["scenes"][scene]["methods"]["gray"]["psnr"],
                "gray_matched": r[baseline_key]["psnr"],
                "noncentral": r["gray-non-central"]["psnr"],
                "spags": r["SPaGS"]["psnr"],
            },
            "rings": {
                "gray_matched": round_list(r[baseline_key]["rings"], 5),
                "noncentral": round_list(r["gray-non-central"]["rings"], 5),
                "spags": round_list(r["SPaGS"]["rings"], 5),
            },
            "per_view": {
                "gray_matched": round_list(r[baseline_key]["per_view"], 5),
                "noncentral": round_list(r["gray-non-central"]["per_view"], 5),
            },
        }
        if scene in rungs:
            entry["rungs"] = {
                name: {"psnr": v["psnr"], "rings": round_list(v["rings"], 5)}
                for name, v in rungs[scene]["rungs"].items()
            }
        payload["scenes"][scene] = entry

    dest = os.path.join(HERE, "report_data.json")
    with open(dest, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(f"wrote {dest}  ({os.path.getsize(dest)/1024:.0f} KB)")

    means = {}
    for key in ("gray_published", "gray_matched", "noncentral", "spags"):
        means[key] = float(np.mean([payload["scenes"][s]["psnr"][key] for s in SCENES]))
    print("means:", {k: round(v, 3) for k, v in means.items()})


if __name__ == "__main__":
    main()
