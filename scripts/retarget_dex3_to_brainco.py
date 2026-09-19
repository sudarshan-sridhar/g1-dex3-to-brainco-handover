r"""Dex3 -> BrainCo Revo2 retargeting layer (assignment items 1.3 correspondence and 1.4 retargeting).

Pure numpy, no Isaac Sim import. Works on the 28-D Stage A recordings written by
scripts/handover_demos.py:

    28-D = [ arms(14) | hands(14) ]        26-D = [ arms(14) | brainco hands(12) ]

Both source blocks are "robot order": the order the joints appear in
``robot.data.joint_names`` of the G1 Dex3 articulation (g1.usd), filtered to the arm
joints (shoulder/elbow/wrist) and to the ``*_hand_*`` joints respectively. Isaac Lab /
PhysX orders joints breadth-first over the kinematic tree and interleaves left/right
at each depth, so the 14 arm joints are

    L_shoulder_pitch, R_shoulder_pitch, L_shoulder_roll, R_shoulder_roll, L_shoulder_yaw,
    R_shoulder_yaw, L_elbow, R_elbow, L_wrist_roll, R_wrist_roll, L_wrist_pitch,
    R_wrist_pitch, L_wrist_yaw, R_wrist_yaw

and the 14 hand joints are the DEX3_HAND_ORDER list below (index_0/middle_0/thumb_0 for
both sides, then index_1/middle_1/thumb_1, then thumb_2). This matches the order the simulator
reports for the G1 + Dex3 articulation. Everything in this file resolves joints BY NAME
through that list, so if the logged order differs, either fix the list or pass
``--hand-order`` / ``--joint-names-json`` to the CLI. The arm block is passed through
untouched, so its exact order never matters here.

Target 26-D hand block = ``brainco_cfg.BRAINCO_ACTUATED`` order (left 6 then right 6,
thumb_metacarpal, thumb_proximal, index, middle, ring, pinky). The BrainCo articulation
in Isaac Lab will again be breadth-first / interleaved, so the BrainCo replay script must
map the 12 columns by name, not positionally.

Retarget rule (normalised closure): for every Dex3 joint
    c = (q - q_open) / (q_closed - q_open),  clipped to [0, 1]
with q_open = 0 (the demo start pose) and q_closed = grip * soft-limit end with the
larger travel from zero, the closed target the scripted Dex3 demos
drive to (grip 0.9, soft_joint_pos_limit_factor 0.9). The BrainCo
target is  lower + c * (upper - lower)  with the official-URDF limits (lower is 0 on all
12 joints). Distal (mimic) joints are not part of the 12-D command; ``apply_mimic``
in brainco_cfg.py fills them at write time.

Sign conventions. Dex3 encodes left/right mirroring in the joint limits (left index/middle
close toward negative q, right toward positive; left thumb_2 closes positive, right
negative), which is why closure is computed against each joint's OWN closed value and
never against a global sign. BrainCo Revo2 (official URDF, verified in
g1_29dof_mode_15_brainco_hand.urdf: right_thumb_metacarpal_joint origin
rpy="3.1416 0 2.9637" vs left rpy="0 0 0.16787", same axis "0 0 1"; all limits 0..upper)
mirrors by joint ORIGIN, not by axis sign, so positive q closes on both sides and no
sign flip is needed anywhere in the BrainCo direction. (The unofficial revo2 merge
mirrors by axis sign instead; do not use its limits/signs with this file.)

CLI
    python scripts/retarget_dex3_to_brainco.py --self-test
    python scripts/retarget_dex3_to_brainco.py --in results/handover_dex3 [--out results/handover_dex3_brainco]
    python scripts/retarget_dex3_to_brainco.py --write-correspondence   # regenerates docs/correspondence.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
CLOSURE_SCALE = 0.9  # full closure maps to 90% of the hard limit, matching the Dex3 grip=0.9 convention and Isaac soft limits

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

# ----------------------------------------------------------------------------------
# Dex3 (source) facts
# ----------------------------------------------------------------------------------
# Expected robot.data.joint_names order of the 14 Dex3 hand joints on g1.usd (see
# module docstring for the caveat). Names measured from results/usd/dex3_probe.json.
DEX3_HAND_ORDER = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
]

# Hard URDF limits. LEFT values back-computed from the soft limits printed by
# the simulator (Isaac Lab soft-limit factor 0.9 about the
# midpoint): e.g. index_0 soft [-1.492, -0.079] -> hard [-1.5708, 0]. RIGHT values are
# the Unitree dex3-1 right-hand URDF (sign-mirrored).
DEX3_LIMITS = {
    "left_hand_thumb_0_joint": (-1.0472, 1.0472),
    "left_hand_thumb_1_joint": (-0.7245, 1.0472),
    "left_hand_thumb_2_joint": (0.0, 1.7453),
    "left_hand_index_0_joint": (-1.5708, 0.0),
    "left_hand_index_1_joint": (-1.7453, 0.0),
    "left_hand_middle_0_joint": (-1.5708, 0.0),
    "left_hand_middle_1_joint": (-1.7453, 0.0),
    "right_hand_thumb_0_joint": (-1.0472, 1.0472),
    "right_hand_thumb_1_joint": (-1.0472, 0.7245),
    "right_hand_thumb_2_joint": (-1.7453, 0.0),
    "right_hand_index_0_joint": (0.0, 1.5708),
    "right_hand_index_1_joint": (0.0, 1.7453),
    "right_hand_middle_0_joint": (0.0, 1.5708),
    "right_hand_middle_1_joint": (0.0, 1.7453),
}
SOFT_LIMIT_FACTOR = 0.9   # G1_29DOF_CFG.soft_joint_pos_limit_factor
DEFAULT_GRIP = 0.9        # Dex3 demo --grip default

# Role dictionary from the scripted Dex3 demos (DEX3_CLOSED). Only the
# KEYS are used here (role identification); the numeric values were superseded in
# those scripts by the soft-limit closure that dex3_closed() reproduces.
DEX3_CLOSED_ROLES = {
    "thumb_0": 0.6, "thumb_1": 0.7, "thumb_2": 0.6,
    "index_0": 0.9, "index_1": 0.9,
    "middle_0": 0.9, "middle_1": 0.9,
}

# ----------------------------------------------------------------------------------
# BrainCo (target) facts, from brainco_cfg.py (official URDF)
# ----------------------------------------------------------------------------------
try:
    sys.path.insert(0, _HERE)
    from brainco_cfg import BRAINCO_ACTUATED, BRAINCO_UPPER  # type: ignore
except Exception:  # brainco_cfg imports isaaclab at module level; fall back to the same constants
    BRAINCO_ACTUATED = [
        f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("thumb_metacarpal", "thumb_proximal", "index_proximal",
                  "middle_proximal", "ring_proximal", "pinky_proximal")
    ]
    BRAINCO_UPPER = {
        "thumb_metacarpal": 1.5184, "thumb_proximal": 1.0472, "thumb_distal": 1.0472,
        "index_proximal": 1.4661, "index_distal": 1.693,
        "middle_proximal": 1.4661, "middle_distal": 1.693,
        "ring_proximal": 1.4661, "ring_distal": 1.693,
        "pinky_proximal": 1.4661, "pinky_distal": 1.693,
    }

BRAINCO_LIMITS = {
    n: (0.0, BRAINCO_UPPER[n.split("_", 1)[1].rsplit("_joint", 1)[0]]) for n in BRAINCO_ACTUATED
}

# BrainCo actuated joint (suffix) <- list of Dex3 joint roles whose closure is averaged.
HAND_JOINT_MAP = {
    "thumb_metacarpal": ["thumb_0"],
    "thumb_proximal": ["thumb_1", "thumb_2"],
    "index_proximal": ["index_0", "index_1"],
    "middle_proximal": ["middle_0", "middle_1"],
    "ring_proximal": ["middle_0", "middle_1"],
    "pinky_proximal": ["middle_0", "middle_1"],
}
# Inverse: BrainCo joint that a Dex3 role reads back from (12 -> 14).
INVERSE_MAP = {
    "thumb_0": "thumb_metacarpal", "thumb_1": "thumb_proximal", "thumb_2": "thumb_proximal",
    "index_0": "index_proximal", "index_1": "index_proximal",
    "middle_0": "middle_proximal", "middle_1": "middle_proximal",
}

N_ARM = 14
N_DEX3_HAND = 14
N_BRAINCO_HAND = 12

# ----------------------------------------------------------------------------------
# Requirement 1.3: correspondence table
# ----------------------------------------------------------------------------------
CORRESPONDENCE = [
    # ---- joints ----
    {"category": "joints", "source": "{side}_hand_thumb_0_joint (Dex3 thumb yaw / abduction)",
     "target": "{side}_thumb_metacarpal_joint (z axis, 0..1.5184)",
     "rule": "closure fraction c of thumb_0 -> 0 + c*1.5184",
     "note": "1:1. Dex3 thumb_0 has symmetric limits +-1.0472; closed = +grip*soft_hi on both sides (handover build_hand picks the end with larger |travel|, ties go to hi)."},
    {"category": "joints", "source": "{side}_hand_thumb_1_joint + {side}_hand_thumb_2_joint (thumb flexion, 2 DOF)",
     "target": "{side}_thumb_proximal_joint (x axis, 0..1.0472); thumb_distal follows x1.0 via apply_mimic",
     "rule": "c = mean(c_thumb_1, c_thumb_2) -> c*1.0472",
     "note": "2 -> 1 DOF. Left thumb_1 closes +, thumb_2 closes +; right thumb_1 closes +0.7245 side, thumb_2 closes negative. Closure is per-joint so the sign difference is absorbed."},
    {"category": "joints", "source": "{side}_hand_index_0_joint + {side}_hand_index_1_joint",
     "target": "{side}_index_proximal_joint (y axis, 0..1.4661); index_distal follows x1.155",
     "rule": "c = mean(c_index_0, c_index_1) -> c*1.4661",
     "note": "2 -> 1 DOF (Dex3 has an independent distal joint, Revo2 couples it)."},
    {"category": "joints", "source": "{side}_hand_middle_0_joint + {side}_hand_middle_1_joint",
     "target": "{side}_middle_proximal_joint AND {side}_ring_proximal_joint AND {side}_pinky_proximal_joint (each 0..1.4661, distal x1.155)",
     "rule": "c = mean(c_middle_0, c_middle_1) -> c*1.4661 written to middle, ring, pinky",
     "note": "Dex3 has no ring/pinky; the three ulnar fingers move as one so the Revo2 palm closes as a unit like the Dex3 middle finger. Dex3 left index/middle close toward negative q, right toward positive; BrainCo closes positive on both sides."},
    {"category": "joints", "source": "(none)", "target": "{side}_{finger}_distal_joint x5 per side",
     "rule": "software mimic: distal = ratio * proximal (thumb 1.0, fingers 1.155), clipped to distal upper",
     "note": "Not in the 26-D action; filled by brainco_cfg.apply_mimic at write time."},
    # ---- fingertips ----
    {"category": "fingertips", "source": "{side}_hand_thumb_2_link (Dex3 distal thumb body; no separate tip frame)",
     "target": "{side}_thumb_tip_Link (left) / right_thumb_tip (right)",
     "rule": "compare world positions, palm-relative", "note": "Dex3 body names in Isaac Lab are the URDF link names; the USD prim path prefix (left_hand_left_hand_*) is the converter's node name, not the body name."},
    {"category": "fingertips", "source": "{side}_hand_index_1_link", "target": "{side}_index_tip_Link / right_index_tip",
     "rule": "as above", "note": ""},
    {"category": "fingertips", "source": "{side}_hand_middle_1_link", "target": "{side}_middle_tip_Link / right_middle_tip, plus ring_tip and pinky_tip (reported against the same Dex3 middle tip)",
     "rule": "as above", "note": "Ring/pinky errors are informational: they have no Dex3 counterpart."},
    # ---- palm / wrist ----
    {"category": "palm pose", "source": "{side}_hand_palm_link, fixed child of {side}_wrist_yaw_link",
     "target": "{side}_base_link, fixed child of {side}_wrist_yaw_link via {side}_base_joint: xyz (0.0591, 0, 0), rpy (-pi/2, 0, -pi/2) left / (+pi/2, 0, +pi/2) right; {side}_base2_link at xyz (0.0415, 0, 0)",
     "rule": "palm frame := wrist_yaw_link frame on both robots (identity), the fixed palm offset is absorbed into the grasp offset (--grasp-offset in handover_demos.py)",
     "note": "The Dex3 palm offset from wrist_yaw_link is not read here (g1.usd only); the 0.0591 m Revo2 mount plus a longer palm means the BrainCo fingers reach ~2-4 cm further along wrist x. Stage B keeps the Dex3 wrist targets unchanged to expose exactly this mismatch."},
    {"category": "wrist pose", "source": "{side}_wrist_yaw_link (IK end-effector in handover_demos.py)",
     "target": "{side}_wrist_yaw_link (identical link name, identical 29-DOF body chain)",
     "rule": "pass-through", "note": "All 29 body joint names, axes and limits are byte-identical between g1.usd and the BrainCo URDF (docs/hand_models.md)."},
    # ---- contacts ----
    {"category": "contacts", "source": "right hand grasp: thumb_2 tip vs index_1 + middle_1 tips (3-point pinch, thumb opposes two fingers)",
     "target": "right hand grasp: right_thumb_tip vs right_index_tip + right_middle_tip (primary), ring/pinky tips wrap as secondary support",
     "rule": "same opposition pairs; ring/pinky are extra contacts", "note": "Grasp phase in the handover: right hand holds from the outer side (grasp offset +x), palm normal +y_base."},
    {"category": "contacts", "source": "left hand grasp: thumb_2 tip vs index_1 + middle_1 tips",
     "target": "left_thumb_tip_Link vs left_index_tip_Link + left_middle_tip_Link (primary), ring/pinky secondary",
     "rule": "same", "note": "Transfer phase: left closes before right opens; the two-hand contact set is the union of the two rows above."},
    {"category": "contacts", "source": "unintended: palm_link, *_0 links vs table/object",
     "target": "unintended: base_link/base2_link, *_proximal links, ring/pinky links vs table/object",
     "rule": "metric M3 counts any contact outside the grasp pairs", "note": ""},
    # ---- observations / actions ----
    {"category": "observations", "source": "state 28-D: [0:14] arm joint pos (robot order, L/R interleaved), [14:28] Dex3 hand joint pos (DEX3_HAND_ORDER)",
     "target": "state 26-D: [0:14] identical arm block, [14:26] BrainCo actuated joint pos in BRAINCO_ACTUATED order",
     "rule": "retarget_state: arms copied, hands via retarget_hand", "note": "Distal (mimic) joints are not observed; they are a deterministic function of the proximal ones."},
    {"category": "actions", "source": "action 28-D: same layout as state (joint position targets)",
     "target": "action 26-D: same layout as the 26-D state (joint position targets); apply_mimic expands 12 -> 22 hand targets at write time",
     "rule": "retarget_action: identical to retarget_state", "note": "Position control on both robots; BrainCo PD gains in brainco_cfg.py (stiffness 10, damping 0.5)."},
]


# ----------------------------------------------------------------------------------
# Core maths
# ----------------------------------------------------------------------------------
def _role(name: str) -> str:
    m = re.search(r"(thumb|index|middle)_(\d)", name)
    if not m:
        raise ValueError(f"not a Dex3 hand joint: {name}")
    return f"{m.group(1)}_{m.group(2)}"


def _side(name: str) -> str:
    return "left" if name.startswith("left") else "right"


def dex3_soft_limits(name: str, factor: float = SOFT_LIMIT_FACTOR) -> tuple[float, float]:
    lo, hi = DEX3_LIMITS[name]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
    return mid - half, mid + half


def dex3_closed(name: str, grip: float = DEFAULT_GRIP) -> float:
    """Closed target used by the Stage A recorder: grip x the soft-limit end with more travel from 0."""
    lo, hi = dex3_soft_limits(name)
    return hi * grip if abs(hi) >= abs(lo) else lo * grip


def dex3_open(name: str) -> float:
    return 0.0


def closure(q: np.ndarray, name: str, grip: float = DEFAULT_GRIP) -> np.ndarray:
    """Closure fraction in [0, 1] for one Dex3 joint (works on scalars or arrays)."""
    qo, qc = dex3_open(name), dex3_closed(name, grip)
    return np.clip((np.asarray(q, dtype=np.float64) - qo) / (qc - qo), 0.0, 1.0)


def retarget_hand(dex3_hand_14: np.ndarray, hand_order: list[str] | None = None,
                  grip: float = DEFAULT_GRIP, return_closure: bool = False) -> np.ndarray:
    """(..., 14) Dex3 hand joint positions -> (..., 12) BrainCo actuated targets.

    Column order in = ``hand_order`` (default DEX3_HAND_ORDER); out = BRAINCO_ACTUATED.
    """
    order = hand_order or DEX3_HAND_ORDER
    q = np.asarray(dex3_hand_14, dtype=np.float64)
    if q.shape[-1] != N_DEX3_HAND or len(order) != N_DEX3_HAND:
        raise ValueError(f"expected last dim {N_DEX3_HAND}, got {q.shape} / order len {len(order)}")
    c = {}  # (side, role) -> closure array
    for k, name in enumerate(order):
        c[(_side(name), _role(name))] = closure(q[..., k], name, grip)
    out = np.zeros(q.shape[:-1] + (N_BRAINCO_HAND,), dtype=np.float64)
    cl = np.zeros_like(out)
    for j, bname in enumerate(BRAINCO_ACTUATED):
        side = _side(bname)
        key = bname.split("_", 1)[1].rsplit("_joint", 1)[0]
        cj = np.mean([c[(side, r)] for r in HAND_JOINT_MAP[key]], axis=0)
        lo, hi = BRAINCO_LIMITS[bname]
        out[..., j] = lo + cj * (hi - lo) * CLOSURE_SCALE   # positive closes on both sides (origin-mirrored URDF)
        cl[..., j] = cj
    return (out, cl) if return_closure else out


def inverse_retarget_hand(brainco_hand_12: np.ndarray, hand_order: list[str] | None = None,
                          grip: float = DEFAULT_GRIP) -> np.ndarray:
    """(..., 12) BrainCo actuated targets -> (..., 14) Dex3 joint positions (closure-preserving)."""
    order = hand_order or DEX3_HAND_ORDER
    b = np.asarray(brainco_hand_12, dtype=np.float64)
    if b.shape[-1] != N_BRAINCO_HAND:
        raise ValueError(f"expected last dim {N_BRAINCO_HAND}, got {b.shape}")
    bidx = {n: i for i, n in enumerate(BRAINCO_ACTUATED)}
    out = np.zeros(b.shape[:-1] + (N_DEX3_HAND,), dtype=np.float64)
    for k, name in enumerate(order):
        side, role = _side(name), _role(name)
        bname = f"{side}_{INVERSE_MAP[role]}_joint"
        lo, hi = BRAINCO_LIMITS[bname]
        cj = np.clip((b[..., bidx[bname]] - lo) / (hi - lo), 0.0, 1.0)
        out[..., k] = dex3_open(name) + cj * (dex3_closed(name, grip) - dex3_open(name))
    return out


def retarget_state(state_28: np.ndarray, hand_order: list[str] | None = None,
                   grip: float = DEFAULT_GRIP) -> np.ndarray:
    s = np.asarray(state_28, dtype=np.float64)
    if s.shape[-1] != N_ARM + N_DEX3_HAND:
        raise ValueError(f"expected last dim 28, got {s.shape}")
    return np.concatenate([s[..., :N_ARM], retarget_hand(s[..., N_ARM:], hand_order, grip)], axis=-1)


def retarget_action(action_28: np.ndarray, hand_order: list[str] | None = None,
                    grip: float = DEFAULT_GRIP) -> np.ndarray:
    """Actions are joint position targets in the same layout as states, so the map is identical."""
    return retarget_state(action_28, hand_order, grip)


# Dex3 fingertip body -> BrainCo fingertip body, per side (metric M2).
FINGERTIP_PAIRS = {
    "left": {"thumb": ("left_hand_thumb_2_link", "left_thumb_tip_Link"),
             "index": ("left_hand_index_1_link", "left_index_tip_Link"),
             "middle": ("left_hand_middle_1_link", "left_middle_tip_Link"),
             "ring": ("left_hand_middle_1_link", "left_ring_tip_Link"),
             "pinky": ("left_hand_middle_1_link", "left_pinky_tip_Link")},
    "right": {"thumb": ("right_hand_thumb_2_link", "right_thumb_tip"),
              "index": ("right_hand_index_1_link", "right_index_tip"),
              "middle": ("right_hand_middle_1_link", "right_middle_tip"),
              "ring": ("right_hand_middle_1_link", "right_ring_tip"),
              "pinky": ("right_hand_middle_1_link", "right_pinky_tip")},
}


def fingertip_tracking_error(dex3_tips: dict, brainco_tips: dict,
                             dex3_palm: np.ndarray | None = None,
                             brainco_palm: np.ndarray | None = None,
                             side: str | None = None) -> dict:
    """Jacobian-free per-finger tracking error (m) for metric M2.

    ``dex3_tips`` / ``brainco_tips``: {finger: xyz} or {body_name: xyz} world positions for
    one side (``side``) or both (keys prefixed by side, e.g. "left/thumb"). If palm
    positions are given, errors are computed palm-relative so the wrist error (metric M2
    wrist part) is separated from the finger geometry error. Ring/pinky are measured
    against the Dex3 middle tip and flagged ``informational``.
    Returns {"per_finger": {...}, "mean": float, "mean_primary": float}.
    """
    def _get(d, side_, finger, body):
        for key in (f"{side_}/{finger}", finger, body):
            if key in d:
                return np.asarray(d[key], dtype=np.float64)
        return None

    sides = [side] if side else ["left", "right"]
    per, prim = {}, []
    for sd in sides:
        for finger, (dbody, bbody) in FINGERTIP_PAIRS[sd].items():
            pd, pb = _get(dex3_tips, sd, finger, dbody), _get(brainco_tips, sd, finger, bbody)
            if pd is None or pb is None:
                continue
            if dex3_palm is not None:
                pd = pd - np.asarray(dex3_palm, dtype=np.float64)
            if brainco_palm is not None:
                pb = pb - np.asarray(brainco_palm, dtype=np.float64)
            err = float(np.linalg.norm(pb - pd))
            per[f"{sd}/{finger}"] = {"error_m": err, "informational": finger in ("ring", "pinky")}
            if finger not in ("ring", "pinky"):
                prim.append(err)
    allv = [v["error_m"] for v in per.values()]
    return {"per_finger": per,
            "mean": float(np.mean(allv)) if allv else float("nan"),
            "mean_primary": float(np.mean(prim)) if prim else float("nan")}


# ----------------------------------------------------------------------------------
# Correspondence document (docs/correspondence.md)
# ----------------------------------------------------------------------------------
def correspondence_markdown() -> str:
    lines = [
        "# 1.3 Source-to-target correspondence: Dex3 to Brainco Revo2",
        "",
        "Generated by `scripts/retarget_dex3_to_brainco.py --write-correspondence` from its `CORRESPONDENCE` table.",
        "`{side}` stands for `left` / `right`. Sources: `docs/hand_models.md` (official URDF), "
        "`scripts/brainco_cfg.py`, `scripts/handover_demos.py` (28-D layout), and the joint names and "
        "soft limits the simulator reports for the G1 + Dex3 articulation.",
        "",
        "## Layouts",
        "",
        "| vector | dims | block 0 | block 1 |",
        "|---|---|---|---|",
        "| Dex3 state / action | 28 | `[0:14]` arm joint pos, robot order (L/R interleaved: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw) | `[14:28]` Dex3 hand joints, robot order (`DEX3_HAND_ORDER`) |",
        "| BrainCo state / action | 26 | `[0:14]` identical arm block | `[14:26]` `BRAINCO_ACTUATED` order (L: thumb_metacarpal, thumb_proximal, index, middle, ring, pinky; then R) |",
        "",
        "`DEX3_HAND_ORDER` (expected PhysX breadth-first order, to be confirmed against the `HANDOVER_HAND_L/R` log line): "
        + ", ".join(f"`{n}`" for n in DEX3_HAND_ORDER),
        "",
        "## Closure rule",
        "",
        "For each Dex3 joint `c = clip((q - 0) / (q_closed - 0), 0, 1)` with `q_closed = grip * soft-limit end with more travel` "
        "(grip 0.9, soft factor 0.9, i.e. what `handover_demos.py build_hand` drives to). BrainCo target `= 0 + c * upper` "
        "(all 12 actuated joints have lower 0). Grouped joints average their closures. No sign flips: Dex3 mirrors L/R by limit sign "
        "(handled per joint), the official Revo2 URDF mirrors by joint origin (`right_thumb_metacarpal_joint` rpy `3.1416 0 2.9637`, "
        "same axis `0 0 1`) so positive closes on both sides.",
        "",
        "| Dex3 joint | hard limits | soft limits | q_open | q_closed (grip 0.9) |",
        "|---|---|---|---|---|",
    ]
    for n in DEX3_HAND_ORDER:
        lo, hi = DEX3_LIMITS[n]
        slo, shi = dex3_soft_limits(n)
        lines.append(f"| `{n}` | [{lo:+.4f}, {hi:+.4f}] | [{slo:+.3f}, {shi:+.3f}] | 0 | {dex3_closed(n):+.3f} |")
    lines += ["", "| BrainCo joint | limits | closed (c=1) |", "|---|---|---|"]
    for n in BRAINCO_ACTUATED:
        lo, hi = BRAINCO_LIMITS[n]
        lines.append(f"| `{n}` | [{lo:.1f}, {hi:.4f}] | {hi:.4f} |")
    lines += ["", "## Correspondence table", "", "| category | source (Dex3, g1.usd) | target (BrainCo Revo2, official URDF) | rule | note |", "|---|---|---|---|---|"]
    for r in CORRESPONDENCE:
        row = [r["category"], r["source"], r["target"], r["rule"], r["note"]]
        lines.append("| " + " | ".join(x.replace("|", "\\|") for x in row) + " |")
    lines += ["", "Right-hand Dex3 hard limits are the sign-mirrored Unitree dex3-1 values and have not yet been confirmed from a log line; "
              "left-hand values are confirmed (DEX3R_LIMIT).", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------
def _stats(arr: np.ndarray, names: list[str]) -> dict:
    return {n: {"min": float(arr[..., i].min()), "max": float(arr[..., i].max())} for i, n in enumerate(names)}


def _load_hand_order(args) -> list[str]:
    if args.hand_order:
        order = [s.strip() for s in args.hand_order.split(",")]
    elif args.joint_names_json:
        d = json.load(open(args.joint_names_json))
        names = d.get("joint_names", d) if isinstance(d, dict) else d
        order = [n for n in names if "_hand_" in n]
    else:
        return DEX3_HAND_ORDER
    if len(order) != N_DEX3_HAND or set(order) != set(DEX3_HAND_ORDER):
        raise SystemExit(f"hand order must be a permutation of the 14 Dex3 hand joints, got {order}")
    return order


def run_batch(args) -> None:
    order = _load_hand_order(args)
    out_dir = args.out or (args.inp.rstrip("/\\") + "_brainco")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.inp, "ep*_actions.npy")) + glob.glob(os.path.join(args.inp, "ep*_states.npy")))
    if not files:
        raise SystemExit(f"no ep*_actions.npy / ep*_states.npy in {args.inp}")
    dex_names = [f"arm_{i}" for i in range(N_ARM)] + order
    bc_names = [f"arm_{i}" for i in range(N_ARM)] + BRAINCO_ACTUATED
    report = {"in": args.inp, "out": out_dir, "grip": args.grip, "hand_order": order,
              "brainco_order": BRAINCO_ACTUATED, "files": {}, "clip_counts": {}}
    agg_in, agg_out = [], []
    for f in files:
        a = np.load(f)
        if a.ndim != 2 or a.shape[1] != N_ARM + N_DEX3_HAND:
            print(f"SKIP {f}: shape {a.shape}", flush=True); continue
        b = retarget_action(a, order, args.grip)
        # clipping: closure outside [0,1] before clip
        clips = {}
        for k, n in enumerate(order):
            qc = dex3_closed(n, args.grip)
            raw = a[:, N_ARM + k] / qc
            clips[n] = int(np.sum((raw < 0.0) | (raw > 1.0)))
        name = os.path.basename(f)
        np.save(os.path.join(out_dir, name), b.astype(np.float32))
        report["files"][name] = {"T": int(a.shape[0]), "in_shape": list(a.shape), "out_shape": list(b.shape),
                                 "hand_clip_counts": clips}
        agg_in.append(a); agg_out.append(b)
    A, B = np.concatenate(agg_in), np.concatenate(agg_out)
    report["per_joint_before"] = _stats(A, dex_names)
    report["per_joint_after"] = _stats(B, bc_names)
    report["clip_counts"] = {n: int(sum(v["hand_clip_counts"][n] for v in report["files"].values())) for n in order}
    report["total_steps"] = int(A.shape[0])
    for extra in ("handover_summary.json",):
        src = os.path.join(args.inp, extra)
        if os.path.exists(src):
            import shutil; shutil.copy(src, os.path.join(out_dir, extra))
    with open(os.path.join(out_dir, "retarget_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"RETARGET wrote {len(report['files'])} files, {A.shape[0]} steps -> {out_dir}", flush=True)
    print(f"RETARGET clip_counts {report['clip_counts']}", flush=True)


def run_self_test(args) -> None:
    order = _load_hand_order(args)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    open_v = np.zeros(N_DEX3_HAND)
    closed_v = np.array([dex3_closed(n, args.grip) for n in order])
    half_v = 0.5 * closed_v
    print("DEX3 order  :", order)
    print("BRAINCO order:", BRAINCO_ACTUATED)
    for label, v in (("open", open_v), ("half", half_v), ("closed", closed_v)):
        out, c = retarget_hand(v, order, args.grip, return_closure=True)
        back = inverse_retarget_hand(out, order, args.grip)
        print(f"\n[{label}] dex3_14   {v}")
        print(f"[{label}] closure12 {c}")
        print(f"[{label}] brainco12 {out}")
        print(f"[{label}] inverse14 {back}   roundtrip_err {np.abs(back - v).max():.2e}")
    # asymmetric: left half closed, right open; thumb-only
    v = np.zeros(N_DEX3_HAND)
    for k, n in enumerate(order):
        if n.startswith("left"):
            v[k] = 0.5 * dex3_closed(n, args.grip)
    print("\n[left-half/right-open] brainco12", retarget_hand(v, order, args.grip))
    v = np.zeros(N_DEX3_HAND)
    for k, n in enumerate(order):
        if "thumb" in n:
            v[k] = dex3_closed(n, args.grip)
    print("[thumb-only closed]     brainco12", retarget_hand(v, order, args.grip))
    # out-of-range clipping
    v = 2.0 * closed_v
    print("[2x closed, clipped]    brainco12", retarget_hand(v, order, args.grip))
    s = np.random.default_rng(0).normal(size=(5, 28)); s[:, 14:] = 0.3 * closed_v
    r = retarget_state(s, order, args.grip)
    assert r.shape == (5, 26) and np.allclose(r[:, :14], s[:, :14]), "arm block must pass through"
    assert np.all(r[:, 14:] >= 0) and np.all(r[:, 14:] <= np.array([BRAINCO_LIMITS[n][1] for n in BRAINCO_ACTUATED]))
    ft = fingertip_tracking_error({"left/thumb": [0, 0, 0], "left/index": [0.01, 0, 0]},
                                  {"left/thumb": [0.005, 0, 0], "left/index": [0.01, 0.02, 0]}, side="left")
    print("fingertip_tracking_error demo:", ft)
    print("\nSELF_TEST_OK")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=str, help="results/<dir> with ep*_actions.npy / ep*_states.npy (T, 28)")
    p.add_argument("--out", type=str, default=None, help="default <in>_brainco")
    p.add_argument("--grip", type=float, default=DEFAULT_GRIP, help="grip used when the source demos were recorded")
    p.add_argument("--hand-order", type=str, default=None, help="comma list of the 14 Dex3 hand joint names in column order")
    p.add_argument("--joint-names-json", type=str, default=None, help="json with 'joint_names' (robot order); hand joints are filtered from it")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--write-correspondence", action="store_true", help="write docs/correspondence.md")
    args = p.parse_args(argv)
    did = False
    if args.write_correspondence:
        dst = os.path.join(_PROJECT, "docs", "correspondence.md")
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(correspondence_markdown())
        print(f"wrote {dst}"); did = True
    if args.self_test:
        run_self_test(args); did = True
    if args.inp:
        run_batch(args); did = True
    if not did:
        p.print_help()


if __name__ == "__main__":
    main()
