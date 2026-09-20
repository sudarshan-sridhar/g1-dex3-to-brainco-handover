"""Merge several handover_demos.py output folders into one folder the LeRobot converter can read.

Episodes are renumbered 0..N-1 across the inputs. Each merged per_episode record keeps its
source folder and source episode id, so every demo stays traceable to the run that made it.

usage: python scripts/merge_demo_dirs.py --drop-black --out data/demos_merged data/demos_dex3/*/
"""
import argparse
import json
import os
import shutil


def is_black(src, i, thresh=5.0):
    """True if any camera video of episode i is black (mean of frame 3 below thresh)."""
    import imageio.v2 as iio
    camdir = os.path.join(src, "cam")
    if not os.path.isdir(camdir):
        return False
    for key in os.listdir(camdir):
        f = os.path.join(camdir, key, f"demo_{i:06d}.mp4")
        if os.path.isfile(f):
            r = iio.get_reader(f)
            try:
                m = float(r.get_data(3).mean())
            finally:
                r.close()
            if m < thresh:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="handover_summary.json")
    ap.add_argument("--drop-black", action="store_true",
                    help="skip episodes whose camera video is black (Isaac Sim 5.1 intermittently starts a "
                         "process with a dead render product; those frames are all zeros)")
    ap.add_argument("srcs", nargs="+")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    merged, base, k = None, None, 0
    for src in a.srcs:
        sp = os.path.join(src, a.summary)
        if not os.path.isfile(sp):
            print(f"skip {src}: no {a.summary}")
            continue
        s = json.load(open(sp))
        if merged is None:
            merged = {kk: vv for kk, vv in s.items() if kk != "per_episode"}
            merged["per_episode"], merged["sources"] = [], []
            base = s
        elif s.get("action_dim") != base.get("action_dim"):
            raise SystemExit(f"{src}: action_dim {s.get('action_dim')} != {base.get('action_dim')}")
        merged["sources"].append({"dir": src, "episodes": len(s["per_episode"]),
                                  "success_rate": s.get("success_rate")})
        for e in s["per_episode"]:
            i = e["episode"]
            if a.drop_black and is_black(src, i):
                merged.setdefault("dropped_black", []).append({"source_dir": src, "source_episode": i})
                continue
            for suf in ("actions", "states", "tips"):
                f = os.path.join(src, f"ep{i}_{suf}.npy")
                if os.path.isfile(f):
                    shutil.copy2(f, os.path.join(a.out, f"ep{k}_{suf}.npy"))
            camdir = os.path.join(src, "cam")
            if os.path.isdir(camdir):
                for key in os.listdir(camdir):
                    f = os.path.join(camdir, key, f"demo_{i:06d}.mp4")
                    if os.path.isfile(f):
                        os.makedirs(os.path.join(a.out, "cam", key), exist_ok=True)
                        shutil.copy2(f, os.path.join(a.out, "cam", key, f"demo_{k:06d}.mp4"))
            r = dict(e)
            r.update({"episode": k, "source_dir": src, "source_episode": i})
            merged["per_episode"].append(r)
            k += 1

    if merged is None:
        raise SystemExit("no inputs")
    n_ok = sum(1 for e in merged["per_episode"] if e.get("success"))
    merged["episodes"] = k
    merged["success_rate"] = n_ok / max(k, 1)
    json.dump(merged, open(os.path.join(a.out, a.summary), "w"), indent=2)
    print(f"MERGED {k} episodes ({n_ok} successful) from {len(merged['sources'])} folders -> {a.out}; "
          f"dropped {len(merged.get('dropped_black', []))} black-camera episodes")


if __name__ == "__main__":
    main()
