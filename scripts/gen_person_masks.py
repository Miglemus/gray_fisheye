#!/usr/bin/env python
"""Generate per-image person masks for FIORD scenes (fisheye + pano spaces).

Pipeline per scene:
  1. Mask2Former (COCO instance, class person) on every stitched pano, in 4
     variants (orig, rot180, yaw-rolled, yaw-rolled+rot180) since the panos are
     upside-down and the photographer often straddles the ERP wrap seam.
  2. Same detector on every registered fisheye image (input_4/cam{1,2}),
     orig + rot180 variants.
  3. Fuse in pano space: pano detections UNION fisheye detections (forward
     projected via colmap poses). Project the fused pano mask back into every
     registered fisheye image and UNION with its direct detection; dilate.
  4. Write:
       <out>/<scene>/pano/<stem>.png            (2048x1024, 255 = person)
       <out>/<scene>/fisheye/cam{1,2}/<img>.png (816x816,   255 = person)
       <out>/<scene>/stats.json                 per-image person fractions
     Raw per-variant detections are cached under <out>/<scene>/_det/ so the
     fusion stage is cheap to re-run.

Run inside /workspace/gray/.venv (transformers + pycolmap + torch).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fiord_mask_geom import CaptureIndex  # noqa: E402

MODEL_ID = "facebook/mask2former-swin-large-coco-instance"
PERSON_LABEL = 0  # COCO class id for person in mask2former label space
SCORE_THR = 0.35
PANO_ROOT = "/workspace/dataset/fiord_pano"
FIORD_BASE = "/workspace/dataset/fiord_baselines"
SCENES = ["bridge_out", "building_in", "corridor_out", "hall_in",
          "kitchen_in", "meetingroom_in", "night_out"]


def load_model(device):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    return processor, model


@torch.no_grad()
def person_mask_batch(processor, model, images_rgb, device):
    """List of HxWx3 uint8 RGB -> list of HxW uint8 (255 = person)."""
    inputs = processor(images=images_rgb, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[im.shape[:2] for im in images_rgb])
    masks = []
    for res, im in zip(results, images_rgb):
        seg, infos = res["segmentation"], res["segments_info"]
        m = np.zeros(im.shape[:2], np.uint8)
        if seg is not None and len(infos):
            seg = seg.cpu().numpy()
            for info in infos:
                if info["label_id"] == PERSON_LABEL and info["score"] >= SCORE_THR:
                    m[seg == info["id"]] = 255
        masks.append(m)
    return masks


def detect_with_variants(processor, model, bgr, variants, device, batch=None):
    """Union of person masks over image variants. variants: list of names."""
    h, w = bgr.shape[:2]
    imgs, undo = [], []
    for var in variants:
        im = bgr
        roll = 0
        if "roll" in var:
            roll = w // 2
            im = np.roll(im, roll, axis=1)
        if "rot180" in var:
            im = im[::-1, ::-1]
        imgs.append(np.ascontiguousarray(im[..., ::-1]))  # BGR->RGB
        undo.append((roll, "rot180" in var))
    masks = person_mask_batch(processor, model, imgs, device)
    out = np.zeros((h, w), np.uint8)
    for m, (roll, rot) in zip(masks, undo):
        if rot:
            m = m[::-1, ::-1]
        if roll:
            m = np.roll(m, -roll, axis=1)
        out = np.maximum(out, m)
    return out


def process_scene(scene, out_root, processor, model, device, sample=0):
    print(f"=== {scene} ===", flush=True)
    idx = CaptureIndex(scene)
    sdir = os.path.join(out_root, scene)
    det_pano = os.path.join(sdir, "_det", "pano")
    det_fish = os.path.join(sdir, "_det", "fisheye")
    os.makedirs(det_pano, exist_ok=True)
    for c in ("cam1", "cam2"):
        os.makedirs(os.path.join(det_fish, c), exist_ok=True)
        os.makedirs(os.path.join(sdir, "fisheye", c), exist_ok=True)
    os.makedirs(os.path.join(sdir, "pano"), exist_ok=True)

    stems = idx.stems()
    if sample:
        stems = stems[:: max(1, len(stems) // sample)][:sample]

    # --- stage 1+2: detections (cached) ---
    for i, stem in enumerate(stems):
        pdet_path = os.path.join(det_pano, stem + ".png")
        if not os.path.exists(pdet_path):
            pano = cv2.imread(os.path.join(idx.pano_dir, "images", stem + ".png"))
            m = detect_with_variants(processor, model, pano,
                                     ["orig", "rot180", "roll", "roll_rot180"],
                                     device)
            cv2.imwrite(pdet_path, m)
        for img in idx.by_stem[stem]:
            name = os.path.basename(img.name)
            fdet_path = os.path.join(det_fish, f"cam{img.camera_id}", name + ".png")
            if os.path.exists(fdet_path):
                continue
            fim = cv2.imread(os.path.join(idx.base, "input_4", img.name))
            if fim is None:
                fim = cv2.imread(os.path.join(
                    idx.base, "images_4", f"cam{img.camera_id}", name))
            m = detect_with_variants(processor, model, fim, ["orig", "rot180"],
                                     device)
            cv2.imwrite(fdet_path, m)
        if (i + 1) % 25 == 0:
            print(f"  det {i + 1}/{len(stems)}", flush=True)

    # --- stage 3: fusion ---
    stats = {"pano": {}, "fisheye": {}}
    kpano = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kfish = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for i, stem in enumerate(stems):
        pano_det = cv2.imread(os.path.join(det_pano, stem + ".png"), 0)
        fish_det = {}
        for img in idx.by_stem[stem]:
            name = os.path.basename(img.name)
            m = cv2.imread(os.path.join(det_fish, f"cam{img.camera_id}",
                                        name + ".png"), 0)
            if m is not None:
                fish_det[name] = m
        fused = np.maximum(pano_det, idx.fisheye_mask_to_pano(stem, fish_det))
        fused = cv2.dilate(fused, kpano)
        cv2.imwrite(os.path.join(sdir, "pano", stem + ".png"), fused)
        stats["pano"][stem] = float((fused > 127).mean())
        for img in idx.by_stem[stem]:
            name = os.path.basename(img.name)
            back = idx.pano_mask_to_fisheye(stem, fused, img)
            m = np.maximum(back, fish_det.get(name, 0))
            m = cv2.dilate(((m > 127) * 255).astype(np.uint8), kfish)
            cv2.imwrite(os.path.join(sdir, "fisheye", f"cam{img.camera_id}",
                                     os.path.splitext(name)[0] + ".png"), m)
            stats["fisheye"][f"cam{img.camera_id}/{name}"] = float((m > 127).mean())
        if (i + 1) % 50 == 0:
            print(f"  fuse {i + 1}/{len(stems)}", flush=True)

    fr = list(stats["fisheye"].values())
    summary = {
        "n_captures": len(stems),
        "mean_person_frac_fisheye": float(np.mean(fr)) if fr else 0.0,
        "frac_images_with_person": float(np.mean([f > 0.001 for f in fr])) if fr else 0.0,
        "mean_person_frac_pano": float(np.mean(list(stats["pano"].values()))) if stats["pano"] else 0.0,
    }
    with open(os.path.join(sdir, "stats.json"), "w") as f:
        json.dump({"summary": summary, **stats}, f, indent=1)
    print(f"[{scene}] {summary}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/dataset/fiord_masks")
    ap.add_argument("--sample", type=int, default=0,
                    help="process only N evenly spaced captures per scene")
    args = ap.parse_args()
    device = "cuda"
    processor, model = load_model(device)
    for scene in args.scenes or SCENES:
        process_scene(scene, args.out, processor, model, device, args.sample)


if __name__ == "__main__":
    main()
