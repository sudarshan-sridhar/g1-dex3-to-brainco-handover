"""Cartesian tracking error at the wrists and fingertips (metric M2).

For every evaluation trial, the commanded joint targets and the joint positions actually reached
are both run through forward kinematics, and the distance between the two is the tracking error.
Because both use the same fixed base, the result does not depend on where the robot stands.

    wrist error     from the 14 arm joints, which are identical on both robots
    fingertip error from the 12 Brainco hand joints (distal joints follow their proximal joint)

The Dex3 hand geometry ships only inside the Isaac Sim USD asset, so the fingertip error is
reported for the Brainco stages; for Stage A the hand column reports the joint-space error instead.

usage: python scripts/tracking_error.py [--urdf <g1_29dof_mode_15_brainco_hand.urdf>]
Writes results/tracking_error.md and results/tracking_error.json
"""
import argparse
import glob
import json
import os

import numpy as np

import brainco_fk as fk_mod

ARM_COLUMNS = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
               "left_shoulder_roll_joint", "right_shoulder_roll_joint",
               "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
               "left_elbow_joint", "right_elbow_joint",
               "left_wrist_roll_joint", "right_wrist_roll_joint",
               "left_wrist_pitch_joint", "right_wrist_pitch_joint",
               "left_wrist_yaw_joint", "right_wrist_yaw_joint"]
HAND_COLUMNS = [f"{side}_{j}_joint" for side in ("left", "right")
                for j in ("thumb_metacarpal", "thumb_proximal", "index_proximal",
                          "middle_proximal", "ring_proximal", "pinky_proximal")]
WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
TIPS = [f"{side}_{f}_tip" for side in ("left", "right")
        for f in ("thumb", "index", "middle", "ring", "pinky")]
CONFIGS = ["stick_30x240", "stick_30x220", "stick_35x240", "heldout_object_round", "heldout_initial_pose"]


def link_name(joints, wanted):
    """The official description spells the left tips with a _Link suffix and the right ones without."""
    for cand in (wanted, wanted + "_Link", wanted + "_link"):
        if cand in joints:
            return cand
    raise KeyError(wanted)


def positions(joints, q_vec, columns, links, base):
    q = dict(zip(columns, q_vec))
    return np.array([fk_mod.fk(joints, l, q, base=base)[:3, 3] for l in links])


def trial_errors(joints, actions, states, with_hands):
    """Mean distance in metres between commanded and reached poses, per trial."""
    n = min(len(actions), len(states)) - 1
    wrist, tip = [], []
    wrist_links = [link_name(joints, w) for w in WRISTS]
    tip_links = [link_name(joints, t) for t in TIPS] if with_hands else []
    for t in range(0, n, 5):     # every 5th control step is plenty for a mean
        cmd, got = actions[t], states[t + 1]
        wrist.append(np.linalg.norm(positions(joints, cmd[:14], ARM_COLUMNS, wrist_links, "pelvis")
                                    - positions(joints, got[:14], ARM_COLUMNS, wrist_links, "pelvis"), axis=1))
        if with_hands:
            tip.append(np.linalg.norm(
                positions(joints, cmd[14:26], HAND_COLUMNS, tip_links, "pelvis")
                - positions(joints, got[14:26], HAND_COLUMNS, tip_links, "pelvis"), axis=1))
    return (float(np.mean(wrist)), float(np.max(wrist)),
            (float(np.mean(tip)), float(np.max(tip))) if with_hands else (None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=fk_mod.URDF)
    a = ap.parse_args()
    joints = fk_mod.load(a.urdf)
    out, rows = {}, ["| stage | configuration | wrist tracking error, mean / max (mm) | fingertip tracking error, mean / max (mm) |",
                     "|---|---|---|---|"]
    for stage in "ABC":
        for cfg in CONFIGS:
            files = sorted(glob.glob(f"results/stage{stage}/{cfg}/*_actions.npy"))
            if not files:
                continue
            w_mean, w_max, t_mean, t_max = [], [], [], []
            for af in files:
                sf = af.replace("_actions.npy", "_states.npy")
                if not os.path.isfile(sf):
                    continue
                wm, wx, (tm, tx) = trial_errors(joints, np.load(af), np.load(sf), with_hands=stage != "A")
                w_mean.append(wm); w_max.append(wx)
                if tm is not None:
                    t_mean.append(tm); t_max.append(tx)
            hands = (f"{1000 * np.mean(t_mean):.1f} / {1000 * np.max(t_max):.1f}" if t_mean
                     else "n/a (Dex3 hand geometry is USD only)")
            rows.append(f"| {stage} | {cfg} | {1000 * np.mean(w_mean):.1f} / {1000 * np.max(w_max):.1f} | {hands} |")
            out[f"{stage}/{cfg}"] = {"wrist_mean_m": float(np.mean(w_mean)), "wrist_max_m": float(np.max(w_max)),
                                     "fingertip_mean_m": float(np.mean(t_mean)) if t_mean else None,
                                     "fingertip_max_m": float(np.max(t_max)) if t_mean else None,
                                     "trials": len(w_mean)}
            print(rows[-1], flush=True)
    header = ("# Cartesian tracking error (M2)\n\n"
              "Distance between the commanded pose and the pose reached one control step later, from\n"
              "forward kinematics on the commanded and the measured joint values. Averaged over every\n"
              "fifth control step of every trial.\n\n")
    open("results/tracking_error.md", "w", encoding="utf-8").write(header + "\n".join(rows) + "\n")
    json.dump(out, open("results/tracking_error.json", "w"), indent=2)
    print("wrote results/tracking_error.md and results/tracking_error.json")


if __name__ == "__main__":
    main()
