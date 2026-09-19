"""Labelled evaluation videos: one mp4 per trial with the stage and outcome burned in.

For every trial in results/stage<S>/<configuration>/ this writes the front and head cameras side
by side (2x upscale) with a banner: stage, configuration, trial number, outcome and a running
step counter. Output: results/videos/stage<S>/<configuration>_trial<NN>_<success|fail>.mp4

usage: python scripts/label_videos.py [--stages A B C] [--scale 2]
"""
import argparse
import glob
import json
import os

import cv2
import imageio.v2 as iio
import numpy as np

STAGE = {"A": "Stage A  |  Original Dex-hand task (Dex3)",
         "B": "Stage B  |  Brainco retargeting only",
         "C": "Stage C  |  Brainco retargeting + fine-tuning"}
CONFIG = {"stick_30x240": "Square stick 3.0 x 24 cm",
          "stick_30x220": "Square stick 3.0 x 22 cm",
          "stick_35x240": "Square stick 3.5 x 24 cm",
          "heldout_object_round": "Held-out object: round stick 3.0 x 24 cm",
          "heldout_initial_pose": "Held-out initial pose"}
FAILURE = {"no_grasp": "no grasp", "drop_before_transfer": "dropped before handover",
           "no_transfer": "handover not completed", "drop_after_transfer": "dropped after handover",
           "no_place": "not placed", "not_released": "not released cleanly",
           "placed_but_propped_or_held": "not released cleanly", "moved_after_place": "moved after placing",
           "final_check_failed": "moved after placing", "fall": "robot fall"}


def banner(frame, lines):
    h, w = frame.shape[:2]
    bh = 24 * len(lines) + 10
    out = np.full((h + bh, w, 3), 24, np.uint8)
    out[bh:] = frame
    for i, (txt, col) in enumerate(lines):
        cv2.putText(out, txt, (10, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)
    return out


def progress(rec, t):
    done = [n for n, k in (("grasped", "t_grasp"), ("handed over", "t_transfer"), ("placed", "t_place"))
            if rec.get(k) is not None and t >= rec[k]]
    return ", ".join(done) if done else "approaching"


def label_trial(stage, cfg, rec_path, trial, out_dir, scale):
    rec = json.load(open(rec_path))
    v = rec.get("videos") or {}
    fr_p, eg_p = v.get("front"), v.get("ego_view")
    if not (fr_p and eg_p and os.path.isfile(fr_p) and os.path.isfile(eg_p)):
        return None
    ok = bool(rec.get("success"))
    outcome = "SUCCESS" if ok else f"FAIL: {FAILURE.get(rec.get('failure'), rec.get('failure'))}"
    color = (90, 220, 110) if ok else (240, 95, 80)
    out_p = os.path.join(out_dir, f"{cfg}_trial{trial:02d}_{'success' if ok else 'fail'}.mp4")
    ra, rb = iio.get_reader(fr_p), iio.get_reader(eg_p)
    w = iio.get_writer(out_p, fps=20, macro_block_size=1)
    for t, (a, b) in enumerate(zip(ra, rb)):
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        w.append_data(banner(np.concatenate([a, b], 1), [
            (f"{STAGE[stage]}  |  {CONFIG.get(cfg, cfg)}  |  trial {trial}", (235, 235, 235)),
            (f"{outcome}  |  t = {t / 20:4.1f} s  |  {progress(rec, t)}  |  front camera, head camera", color)]))
    w.close(); ra.close(); rb.close()
    return out_p, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    for st in a.stages:
        out_dir = f"results/videos/stage{st}"
        os.makedirs(out_dir, exist_ok=True)
        for cfg in CONFIG:
            recs = sorted(glob.glob(f"results/stage{st}/{cfg}/*_ep*.json"))
            done = [label_trial(st, cfg, r, i, out_dir, a.scale) for i, r in enumerate(recs)]
            print(f"stage {st} {cfg}: {len([d for d in done if d])} labelled videos")


if __name__ == "__main__":
    main()
