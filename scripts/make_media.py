"""README media: a side-by-side GIF of the three stages and a still overview.

Uses the front-camera videos of one trial of one configuration for each stage. All three
trials share the same seed, so the stick starts in the same place.

usage: python scripts/make_media.py [--config stick_30x220] [--trial 7]
Needs imageio with the ffmpeg plugin (imageio-ffmpeg) and matplotlib.
"""
import argparse
import glob
import json
import os
import subprocess
import tempfile

import cv2
import imageio.v2 as iio
import numpy as np

TITLES = {"A": ("Stage A", "Dex3, original task"),
          "B": ("Stage B", "Brainco, retargeting only"),
          "C": ("Stage C", "Brainco, retargeting + fine-tuning")}
FAILURE = {"no_grasp": "no grasp", "drop_before_transfer": "dropped", "drop_after_transfer": "dropped",
           "no_transfer": "no handover", "not_released": "not released", "moved_after_place": "moved after placing"}


def trial_record(stage, cfg, trial):
    recs = sorted(glob.glob(f"results/stage{stage}/{cfg}/*_ep*.json"))
    return json.load(open(recs[trial]))


def frames(path):
    return [f for f in iio.get_reader(path)]


def panel(img, stage, status, color, width):
    img = cv2.resize(img, (width, width), interpolation=cv2.INTER_CUBIC)
    bar = np.full((52, width, 3), 250, np.uint8)
    name, sub = TITLES[stage]
    cv2.putText(bar, name, (8, 21), cv2.FONT_HERSHEY_DUPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    out = np.concatenate([bar, img], 0)
    if status:
        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (width - tw - 16, 60), (width - 4, 60 + th + 12), color, -1)
        cv2.putText(out, status, (width - tw - 10, 66 + th), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def status_at(rec, t):
    if rec.get("t_place") is not None and t >= rec["t_place"]:
        return "placed", (46, 139, 87)
    if rec.get("t_transfer") is not None and t >= rec["t_transfer"]:
        return "handed over", (59, 110, 165)
    if rec.get("t_grasp") is not None and t >= rec["t_grasp"]:
        return "grasped", (59, 110, 165)
    return "", (0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stick_30x220")
    ap.add_argument("--trial", type=int, default=7)
    ap.add_argument("--width", type=int, default=260)
    ap.add_argument("--stride", type=int, default=3, help="keep every n-th control step (n x real time)")
    a = ap.parse_args()
    os.makedirs("media", exist_ok=True)
    recs = {s: trial_record(s, a.config, a.trial) for s in "ABC"}
    vids = {s: frames(recs[s]["videos"]["front"]) for s in "ABC"}
    T = max(len(v) for v in vids.values())
    ffmpeg = __import__("imageio_ffmpeg").get_ffmpeg_exe()

    # side-by-side GIF (encoded through a palette so it stays small and clean)
    tmp = os.path.join(tempfile.gettempdir(), "stages.mp4")
    w = iio.get_writer(tmp, fps=20, macro_block_size=1, quality=9)
    for t in range(0, T + 30, a.stride):
        row = []
        for s in "ABC":
            v, rec = vids[s], recs[s]
            k = min(t, len(v) - 1)
            st, col = status_at(rec, k)
            if k == len(v) - 1 and not rec.get("success"):
                st, col = FAILURE.get(rec.get("failure"), "failed"), (200, 85, 61)
            row.append(panel(v[k], s, st, col, a.width))
            row.append(np.full((row[-1].shape[0], 6, 3), 255, np.uint8))
        w.append_data(np.concatenate(row[:-1], 1))
    w.close()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp, "-vf",
                    "fps=20,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    "media/handover_stages.gif"], check=True)
    print("media/handover_stages.gif", round(os.path.getsize("media/handover_stages.gif") / 1e6, 2), "MB")

    # still overview: middle of the task and the end of the trial for each stage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for j, s in enumerate("ABC"):
        rec, v = recs[s], vids[s]
        mid = rec.get("t_transfer") or min(260, len(v) - 1)
        for i, (k, lab) in enumerate(((mid, "handover" if rec.get("t_transfer") else "grasp attempt"),
                                      (len(v) - 1, "end of trial"))):
            ax = axes[i, j]
            ax.imshow(v[min(k, len(v) - 1)])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{lab}, t = {min(k, len(v) - 1) / 20:.1f} s", fontsize=9)
            if i == 0:
                ax.set_title(f"{TITLES[s][0]}\n{TITLES[s][1]}", fontsize=11)
    plt.tight_layout()
    plt.savefig("media/stage_overview.png", dpi=130)
    print("media/stage_overview.png")


if __name__ == "__main__":
    main()
