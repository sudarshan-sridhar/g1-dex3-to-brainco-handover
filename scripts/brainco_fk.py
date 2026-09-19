"""Forward kinematics of the Brainco Revo2 right hand from the official G1 URDF (no simulator).

Used to explain why retargeting the Dex3 closure fails on this hand and what grasp works
instead. The figure it writes compares the retargeted closing motion (thumb swing, thumb bend
and finger curl together) with the opposition grasp used for the Stage C demonstrations.

    python scripts/brainco_fk.py                      # print fingertip positions
    python scripts/brainco_fk.py --figure media/brainco_thumb_opposition.png

Positions are in the right wrist frame, re-expressed in the world axes of the home pose
(x across the body, y forward, z up), which is how the hand sees an upright stick.
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets_brainco", "unitree_ros",
                    "robots", "g1_with_brainco_hand", "g1_29dof_mode_15_brainco_hand.urdf")
BASE = "right_wrist_yaw_link"
TIPS = ["right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip", "right_pinky_tip"]


def load(urdf):
    joints = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        o, a, m = j.find("origin"), j.find("axis"), j.find("mimic")
        joints[j.find("child").get("link")] = dict(
            name=j.get("name"), parent=j.find("parent").get("link"), type=j.get("type"),
            xyz=np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]),
            rpy=np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]),
            axis=np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()]),
            mimic=(m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0))) if m is not None else None)
    return joints


def rot(axis, t):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rpy(r, p, y):
    return rot(np.array([0, 0, 1.0]), y) @ rot(np.array([0, 1.0, 0]), p) @ rot(np.array([1.0, 0, 0]), r)


def fk(joints, link, q, base=BASE):
    """4x4 transform of `link` in `base` for joint values q (name -> rad); mimic joints follow."""
    chain, l = [], link
    while l != base:
        chain.append(joints[l])
        l = joints[l]["parent"]
    T = np.eye(4)
    for jd in reversed(chain):
        A = np.eye(4); A[:3, :3] = rpy(*jd["rpy"]); A[:3, 3] = jd["xyz"]
        v = q.get(jd["name"], 0.0)
        if jd["mimic"]:
            v = q.get(jd["mimic"][0], 0.0) * jd["mimic"][1] + jd["mimic"][2]
        if jd["type"] in ("revolute", "continuous"):
            R = np.eye(4); R[:3, :3] = rot(jd["axis"], v); A = A @ R
        T = T @ A
    return T


def world_axes(p):
    """wrist frame -> world axes at the home pose (x across, y forward, z up)."""
    return np.array([-p[1], p[0], p[2]])


def paths(joints, n=40):
    """Thumb and index fingertip paths (cm, world axes) for the two closing strategies."""
    c = np.linspace(0, 1, n)
    tip = lambda q, l: world_axes(fk(joints, l, q)[:3, 3]) * 100
    # Stage B: the retargeted Dex3 command closes swing, thumb bend and fingers together
    # (closed values of the retargeted grip: swing 1.37, bend 0.94, fingers 1.32 rad).
    retarget = [dict(right_thumb_metacarpal_joint=1.367 * v, right_thumb_proximal_joint=0.943 * v,
                     right_index_proximal_joint=1.32 * v) for v in c]
    # Stage C demonstrations: thumb swung across first, then thumb bend and finger curl close.
    oppose = [dict(right_thumb_metacarpal_joint=1.45, right_thumb_proximal_joint=0.55 * v,
                   right_index_proximal_joint=1.4 * v) for v in c]
    return {name: (np.array([tip(q, "right_thumb_tip") for q in qs]), np.array([tip(q, "right_index_tip") for q in qs]))
            for name, qs in (("retarget", retarget), ("oppose", oppose))}


def figure(joints, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = paths(joints)
    styles = {"retarget": ("#c8553d", "retargeted Dex3 close (Stage B)"),
              "oppose": ("#2e8b57", "opposition grasp (Stage C demonstrations)")}
    fig, (top, side) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, (thumb, index) in P.items():
        col, lab = styles[name]
        top.plot(thumb[:, 0], thumb[:, 1], "-", color=col, lw=2.5, label=f"thumb tip, {lab}")
        side.plot(thumb[:, 1], thumb[:, 2], "-", color=col, lw=2.5)
        if name == "oppose":   # the index curl is the same in both strategies
            top.plot(index[:, 0], index[:, 1], "--", color="#666", lw=1.5, label="index fingertip curl (same in both)")
            side.plot(index[:, 1], index[:, 2], "--", color="#666", lw=1.5)
        for ax, a, b in ((top, 0, 1), (side, 1, 2)):
            ax.plot(thumb[0, a], thumb[0, b], "o", color=col, ms=5)
    top.add_patch(plt.Circle((-5.0, 13.5), 1.5, color="#e0a800", alpha=0.55))
    side.add_patch(plt.Rectangle((12.0, -6.0), 3.0, 18.0, color="#e0a800", alpha=0.35))
    top.text(-5.0, 13.5, "stick", ha="center", va="center", fontsize=8)
    side.text(13.5, 10.5, "stick", ha="center", fontsize=8)
    top.set(xlabel="across the palm (cm)", ylabel="forward from the wrist (cm)", title="Top view")
    side.set(xlabel="forward from the wrist (cm)", ylabel="height above the wrist (cm)",
             title="Side view: where the thumb comes from")
    top.set_aspect("equal")
    for ax in (top, side):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*top.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Brainco Revo2 right hand closing on an upright stick (dots mark the start of each thumb path)",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(out, dpi=160)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()
    joints = load(a.urdf)
    for name, q in (("open", {}), ("closed fingers", {f"right_{f}_proximal_joint": 1.4661 for f in ("index", "middle", "ring", "pinky")})):
        print(name, {t.split("_")[1]: np.round(world_axes(fk(joints, t, q)[:3, 3]), 3).tolist() for t in TIPS})
    if a.figure:
        figure(joints, a.figure)


if __name__ == "__main__":
    main()
