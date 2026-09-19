r"""Convert handover_demos.py recordings into a GR00T-flavoured LeRobot v2 dataset.

Parameterised on the action dimension so the same converter serves both hands:

    --dim 28   Dex3    [arms(14) | hands(14)]   robot-type unitree_g1_dex3_handover
    --dim 26   BrainCo [arms(14) | hands(12)]   robot-type unitree_g1_brainco_handover

Both blocks are INTERLEAVED left/right in the robot's own joint ordering (there is no contiguous
"left_arm" slice). The hands block is
[14:dim]; for BrainCo that is the 12 motor-backed joints of brainco_cfg.BRAINCO_ACTUATED
(left hand then right hand); the distal mimic joints are not part of the action.

FPS: the handover recorder runs at 120 Hz physics / decimation 6 = 20 Hz control,
so the dataset fps is 20. NOTE the
recorder's --video-fps still defaults to 30; record with --video-fps 20 so the mp4 and
the parquet timestamps agree (this script warns when they do not).

INPUT  (from handover_demos.py --out <dir>)
    <in>/ep{i}_actions.npy, ep{i}_states.npy   (T, dim) float32
    <in>/cam/{front,ego_view}/demo_{i:06d}.mp4
    <in>/handover_summary.json                  per_episode[] with the staged metrics

OUTPUT (LeRobot v2)
    <out>/data/chunk-000/episode_XXXXXX.parquet
    <out>/videos/chunk-000/observation.images.<key>/episode_XXXXXX.mp4
    <out>/meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
    <out>/meta/episode_meta.json   per exported episode: source episode id, success,
                                   t_grasp, t_transfer, t_place, dropped, fell,
                                   wrist_track_err, joint_limit_violations, object,
                                   seed (M5 adaptation cost, M6 failure analysis).
"""

import argparse
import json
import os
import shutil
import warnings

import numpy as np

ROBOT_TYPES = ("unitree_g1_dex3_handover", "unitree_g1_brainco_handover")
DEFAULT_ROBOT_TYPE = {28: "unitree_g1_dex3_handover", 26: "unitree_g1_brainco_handover"}
# Copied verbatim from handover_summary.json per_episode[] into meta/episode_meta.json.
EPISODE_META_KEYS = ("object", "t_grasp", "t_transfer", "t_place", "dropped", "fell",
                     "wrist_track_err", "joint_limit_violations", "completion_step",
                     "frames", "success", "obj_start", "obj_end")


def build_modality(video_keys, dim):
    return {
        "state": {"arms": {"start": 0, "end": 14},
                  "hands": {"start": 14, "end": dim}},
        "action": {"arms": {"start": 0, "end": 14},
                   "hands": {"start": 14, "end": dim}},
        "video": {vk: {"original_key": f"observation.images.{vk}"} for vk in video_keys},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def video_meta(path):
    """(height, width, fps) of the first frame; fps None if the reader has no metadata."""
    import imageio.v2 as imageio
    r = imageio.get_reader(path)
    try:
        fr = r.get_data(0)
        fps = None
        try:
            fps = float(r.get_meta_data().get("fps"))
        except Exception:  # noqa: BLE001 -- metadata is best-effort
            pass
    finally:
        r.close()
    return int(fr.shape[0]), int(fr.shape[1]), fps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True,
                    help="results dir written by handover_demos.py --out")
    ap.add_argument("--out", required=True, help="LeRobot dataset dir to create")
    ap.add_argument("--dim", type=int, choices=(28, 26), required=True,
                    help="action/state width: 28 = Dex3, 26 = BrainCo")
    ap.add_argument("--robot-type", choices=ROBOT_TYPES, default=None,
                    help="defaults from --dim")
    ap.add_argument("--task-text", default="hand the object from the right hand to "
                                           "the left hand and place it on the left")
    ap.add_argument("--fps", type=int, default=20,
                    help="control rate of the recorder (120 Hz / decimation 6)")
    ap.add_argument("--keep", choices=("success", "all"), default="success")
    ap.add_argument("--summary", default="handover_summary.json",
                    help="summary file name inside --in")
    ap.add_argument("--action-shift", type=int, default=1,
                    help="pair observation t with action t+shift. handover_demos.py records the state and "
                         "camera frame AFTER applying action t, so the command to learn from observation t "
                         "is action t+1 (shift 1, the last action repeated). 0 reproduces the old pairing, "
                         "which makes a deployed policy replay one executed step per chunk.")
    args = ap.parse_args()
    robot_type = args.robot_type or DEFAULT_ROBOT_TYPE[args.dim]

    import pandas as pd

    summary = json.load(open(os.path.join(args.src, args.summary)))
    per_ep = {e["episode"]: e for e in summary["per_episode"]}
    rec_dim = summary.get("action_dim")
    if rec_dim is not None and rec_dim != args.dim:
        raise SystemExit(f"recorder action_dim {rec_dim} != --dim {args.dim}")
    rec_hz = summary.get("control_hz")
    if rec_hz is not None and abs(rec_hz - args.fps) > 1e-6:
        warnings.warn(f"recorder control_hz {rec_hz} != --fps {args.fps}; the parquet "
                      f"timestamps will not match real time")
    cam_dir = os.path.join(args.src, "cam")
    video_keys = sorted(os.listdir(cam_dir)) if os.path.isdir(cam_dir) else []
    print(f"video keys: {video_keys}")

    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "data", "chunk-000"), exist_ok=True)

    kept, total, eps_meta, ep_meta_out = 0, 0, [], []
    for i in sorted(per_ep):
        if args.keep == "success" and not per_ep[i]["success"]:
            continue
        act = np.load(os.path.join(args.src, f"ep{i}_actions.npy")).astype(np.float32)
        st = np.load(os.path.join(args.src, f"ep{i}_states.npy")).astype(np.float32)
        if act.ndim != 2 or act.shape[1] != args.dim or st.shape[1] != args.dim:
            raise SystemExit(f"ep{i}: expected (T,{args.dim}), got actions {act.shape} "
                             f"states {st.shape}")
        T = min(len(act), len(st))
        act, st = act[:T], st[:T]
        if args.action_shift > 0:        # observation t (after action t) -> next command t+1
            k = args.action_shift
            act = np.concatenate([act[k:], np.repeat(act[-1:], k, axis=0)], axis=0)

        pd.DataFrame({
            "observation.state": [x for x in st],
            "action": [x for x in act],
            "timestamp": np.arange(T, dtype=np.float32) / args.fps,
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, kept, dtype=np.int64),
            "index": np.arange(total, total + T, dtype=np.int64),
            "task_index": np.zeros(T, dtype=np.int64),
        }).to_parquet(os.path.join(args.out, "data", "chunk-000",
                                   f"episode_{kept:06d}.parquet"), index=False)

        for vk in video_keys:
            src = os.path.join(cam_dir, vk, f"demo_{i:06d}.mp4")
            if not os.path.exists(src):
                raise SystemExit(f"ep{i} kept but {src} missing")
            d = os.path.join(args.out, "videos", "chunk-000",
                             f"observation.images.{vk}")
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(src, os.path.join(d, f"episode_{kept:06d}.mp4"))

        eps_meta.append({"episode_index": kept, "tasks": [args.task_text],
                         "length": int(T)})
        m = {"episode_index": kept, "source_episode": int(i), "length": int(T),
             "seed": summary.get("seed")}
        m.update({k: per_ep[i].get(k) for k in EPISODE_META_KEYS})
        ep_meta_out.append(m)
        total += T
        kept += 1

    if kept == 0:
        raise SystemExit("no episodes to export")

    with open(os.path.join(args.out, "meta", "episodes.jsonl"), "w") as f:
        for e in eps_meta:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(args.out, "meta", "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task_text}) + "\n")
    with open(os.path.join(args.out, "meta", "modality.json"), "w") as f:
        json.dump(build_modality(video_keys, args.dim), f, indent=2)
    with open(os.path.join(args.out, "meta", "episode_meta.json"), "w") as f:
        json.dump({"source": os.path.abspath(args.src), "robot_type": robot_type,
                   "keep": args.keep, "fps": args.fps, "action_dim": args.dim,
                   "source_success_rate": summary.get("success_rate"),
                   "source_episodes": summary.get("episodes"),
                   "control_hz": rec_hz, "grip": summary.get("grip"),
                   "pose_noise": summary.get("pose_noise"),
                   "phase_steps": summary.get("phase_steps"),
                   "joint_names": summary.get("joint_names"),
                   "arm_joint_ids": summary.get("arm_joint_ids"),
                   "hand_joint_ids": summary.get("hand_joint_ids"),
                   "per_episode": ep_meta_out}, f, indent=2)

    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": kept, "total_frames": total, "total_tasks": 1,
        "total_chunks": 1, "chunks_size": 1000, "fps": args.fps,
        "splits": {"train": f"0:{kept}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [args.dim]},
            "action": {"dtype": "float32", "shape": [args.dim]},
        },
    }
    if video_keys:
        info["video_path"] = ("videos/chunk-{episode_chunk:03d}/{video_key}/"
                              "episode_{episode_index:06d}.mp4")
        for vk in video_keys:
            h, w, vfps = video_meta(os.path.join(args.out, "videos", "chunk-000",
                                                 f"observation.images.{vk}",
                                                 "episode_000000.mp4"))
            if vfps is not None and abs(vfps - args.fps) > 0.5:
                warnings.warn(f"{vk}: mp4 encoded at {vfps} fps but dataset fps is "
                              f"{args.fps}; re-record with --video-fps {args.fps}")
            info["features"][f"observation.images.{vk}"] = {
                "dtype": "video", "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": float(args.fps), "video.height": h,
                         "video.width": w, "video.channels": 3,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "has_audio": False},
            }
    with open(os.path.join(args.out, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(args.out, "meta", "conversion.json"), "w") as f:
        json.dump({"source": args.src, "action_shift": args.action_shift, "keep": args.keep}, f, indent=2)

    print(f"WROTE {kept}/{len(per_ep)} episodes, {total} frames, dim {args.dim}, "
          f"fps {args.fps}, action_shift {args.action_shift} -> {args.out}")
    print("HANDOVER_CONVERSION_OK")


if __name__ == "__main__":
    main()
