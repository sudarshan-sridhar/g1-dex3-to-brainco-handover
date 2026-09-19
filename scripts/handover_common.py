r"""Shared scene / hand / metrics code for the bimanual handover task on the G1.

IMPORT ORDER: this module imports isaaclab, so it must be imported AFTER
``AppLauncher(args).app`` has been created in the calling script
(scripts/handover_demos.py, scripts/handover_rollout.py). It never parses CLI
arguments itself; the shared CLI flags live in ``handover_common_args.py`` (pure
argparse, importable before AppLauncher) and every function here takes the parsed
``args`` namespace it needs.

Robots
    dex3     G1_29DOF_CFG (g1.usd, Dex3 hands)      state/action 28 = arms(14) | hands(14)
    brainco  G1_BRAINCO_CFG (brainco_cfg.py)        state/action 26 = arms(14) | hands(12)

The arm block is identical on both robots (same 14 joint names, same robot order).
The Dex3 hand block is the 14 ``*_hand_*`` joints in robot order; the BrainCo hand
block is the 12 ``BRAINCO_ACTUATED`` joints in that list's order (mapped by NAME to
articulation indices, never positionally), with the 10 distal joints filled by
``apply_mimic`` every time a target is written.

Scene: one packing table, one green place marker, and ALL THREE object kinds
(steering_wheel, cube, cylinder). Only the active one is placed on the table; the
others are parked far away on the ground. This lets one process run several
configurations (handover_rollout.py --configs) without rebuilding the stage.
"""

import glob
import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Offline re-scoring (pure numpy: defined BEFORE the isaaclab imports so that
# `python scripts/handover_common.py --rescore <episode_json>` works without a
# simulator / Isaac python).
# ---------------------------------------------------------------------------
REST_TOL = 0.03          # |obj_z - resting_z| that still counts as "on the table"
HAND_CLEAR = 0.15        # both wrists must be at least this far from a placed object
PLACE_HOLD = 5           # consecutive steps the place condition must hold


def rescore_record(rec, place_xy=None, place_radius=None, tips=None):
    """Re-score one episode record (the dict written to ep{i}_{stage}.json).

    Applies the FINAL-FRAME re-validation that the live tracker now also applies:
    the object must end inside `place_radius` of the target AND be resting on the
    table (|z - z0| < REST_TOL). Optionally also checks the final wrist clearance
    from ep{i}_tips.npy (rows 0/1 = left/right wrist). Returns a dict.
    """
    z0 = float(rec["z0"])
    obj_end = np.asarray(rec["obj_end"], np.float64)
    pr = float(place_radius if place_radius is not None
               else rec.get("place_radius", 0.12))
    if place_xy is not None:
        d_xy = float(np.linalg.norm(obj_end[:2] - np.asarray(place_xy, np.float64)))
    elif rec.get("place_xy") is not None:
        d_xy = float(np.linalg.norm(obj_end[:2] - np.asarray(rec["place_xy"], np.float64)))
    else:
        d_xy = float(rec["final_dist_to_target_xy"])
    near = d_xy <= pr
    resting = abs(float(obj_end[2]) - z0) < REST_TOL
    out = {"final_dist_to_target_xy": d_xy, "place_radius": pr,
           "final_near_xy": bool(near), "final_dz": round(float(obj_end[2]) - z0, 4),
           "final_resting": bool(resting),
           "t_grasp": rec.get("t_grasp"), "t_transfer": rec.get("t_transfer"),
           "t_place": rec.get("t_place"), "frames": rec.get("frames"),
           "old_success": bool(rec.get("success"))}
    if tips is not None and len(tips):
        f = np.asarray(tips[-1], np.float64)
        out["final_dL"] = round(float(np.linalg.norm(f[0] - obj_end)), 4)
        out["final_dR"] = round(float(np.linalg.norm(f[1] - obj_end)), 4)
        out["final_hands_clear"] = bool(min(out["final_dL"], out["final_dR"]) > HAND_CLEAR)
    # phase gating, when the run recorded its phase boundaries
    pb = rec.get("phase_bounds")
    gate_bad = []
    if pb:
        def first_step_of(cands):
            for nm, lo, _hi in pb:
                if nm in cands:
                    return lo
            return None
        for stage, cands in (("t_grasp", ("r_close", "close_rp")),
                             ("t_transfer", ("l_close",)), ("t_place", ("lower",))):
            lo, t = first_step_of(cands), rec.get(stage)
            if lo is not None and t is not None and t < lo:
                gate_bad.append(f"{stage}={t} fired before its phase (starts at {lo})")
    out["phase_gate_violations"] = gate_bad
    out["success"] = bool(rec.get("t_place") is not None and not rec.get("dropped")
                          and not rec.get("fell") and near and resting and not gate_bad)
    return out


def _rescore_main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="handover_common.py",
                                 description="offline re-score of a handover episode json")
    ap.add_argument("--rescore", type=str, nargs="+", required=True,
                    help="ep*_<stage>.json file(s)")
    ap.add_argument("--place-xy", type=float, nargs=2, default=None)
    ap.add_argument("--place-radius", type=float, default=None)
    ap.add_argument("--write", action="store_true",
                    help="write the corrected success/failure back into the json")
    a = ap.parse_args(argv)
    rc = 0
    for path in a.rescore:
        with open(path) as f:
            rec = json.load(f)
        tips = None
        tp = os.path.join(os.path.dirname(path),
                          f"ep{rec.get('episode', 0)}_tips.npy")
        if os.path.exists(tp):
            tips = np.load(tp)
        r = rescore_record(rec, a.place_xy, a.place_radius, tips)
        print(f"RESCORE {path}")
        for k, v in r.items():
            print(f"  {k}: {v}")
        if a.write:
            rec["success"] = r["success"]
            rec["final_near_xy"] = r["final_near_xy"]
            rec["final_resting"] = r["final_resting"]
            rec["rescored"] = True
            if not r["success"]:
                rec["failure"] = ("moved_after_place" if rec.get("t_place") is not None
                                  else rec.get("failure"))
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"  wrote {path}")
        rc |= 0 if r["success"] else 1
    return 0


if __name__ == "__main__" and "--rescore" in sys.argv:
    raise SystemExit(_rescore_main(sys.argv[1:]))


import torch

import isaaclab.sim as sim_utils
from pxr import UsdPhysics
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from brainco_cfg import (  # noqa: E402
    BRAINCO_ACTUATED, BRAINCO_MIMIC, BRAINCO_UPPER, G1_BRAINCO_CFG, apply_mimic)  # noqa: F401
from retarget_dex3_to_brainco import (  # noqa: E402
    DEX3_HAND_ORDER, FINGERTIP_PAIRS, dex3_closed, inverse_retarget_hand, retarget_hand)

BRAINCO_USD = os.path.join(PROJECT_ROOT, "assets_brainco", "usd", "g1_29dof_brainco.usd")

OBJECT_KINDS = ("steering_wheel", "cube", "cylinder", "bar")
ROBOTS = ("dex3", "brainco")
STATE_DIM = {"dex3": 28, "brainco": 26}
HAND_DIM = {"dex3": 14, "brainco": 12}
N_ARM = 14

DEX3_ROLES = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
ROBOT_POS = (0.0, 0.0, 1.0)
ROBOT_ROT = (0.7071, 0.0, 0.0, 0.7071)      # faces world +y
OBJ_SIZE = 0.06
STEERING_WHEEL_REST_Z = 1.0352               # fallback only
TABLE_TOP_Z = 1.035                          # FALLBACK ONLY - the real top is measured at
                                             # scene setup by measure_table_top() (the packing
                                             # table surface is ~0.994, not 1.035).
TABLE_TOP = None                             # set by measure_table_top(); HANDOVER_TABLE_TOP
CYL_RADIUS, CYL_HEIGHT = 0.03, 0.12
# --- horizontal bar + risers (the object the Dex3 three-finger hand can actually hold) ---
# The bar lies along world x (the robot's left-right), so the right hand takes the +x half
# and the left hand the -x half: the two hands never contest the same volume and never
# cross.  Two static risers under the ENDS lift it RISER_H off the table, so there is
# clearance under the middle of the bar for the fingers to pass, by construction.
BAR_LENGTH, BAR_RADIUS = 0.30, 0.02
RISER_H = 0.05
RISER_SIZE = (0.03, 0.06, RISER_H)           # thin in x so the fingers clear it
BAR_AXIS = "x"                               # set from args.bar_axis by the scene builder
BAR_SHAPE = "cyl"                            # set from args.bar_shape by the scene builder
THUMB_LEAD = 0.0                             # BrainCo closing schedule, set from args.thumb_lead
BC_OPPOSE = None                             # BrainCo-native opposition grasp (m_pre, p_close, f_close), set from args.bc_oppose
BC_SWING = {"L": 1.0, "R": 1.0}              # fraction of m_pre applied per side (the demo ramps it while the hand is clear of the object)
STICK = None                                 # (w, h) when --stick: the "cylinder" kind becomes an upright square stick
RISER_DX = 0.135                             # riser centre offset from the bar centre
TABLE_TOP_PRIOR = 0.9941                     # measured table top; used only to
                                             # place the static risers at scene-build time,
                                             # everything else uses measure_table_top()
OBJ_HALF_Z = {"steering_wheel": 0.0421,      # half height (z) of each object, bbox-measured
              "cube": OBJ_SIZE / 2,          # at scene setup; these are fallbacks
              "cylinder": CYL_HEIGHT / 2,
              "bar": BAR_RADIUS}
WHEEL_RADIUS = 0.143                         # USD bbox 0.286 x 0.286 x 0.084 at scale 0.75
PARK_XY = {"steering_wheel": (3.0, 3.0), "cube": (3.6, 3.0), "cylinder": (4.2, 3.0),
           "bar": (4.8, 3.0)}

# Fingertip bodies per robot and side, in a fixed finger order.
TIP_BODIES = {
    "dex3": {"left": ["left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link"],
             "right": ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]},
    "brainco": {"left": [FINGERTIP_PAIRS["left"][f][1] for f in ("thumb", "index", "middle", "ring", "pinky")],
                "right": [FINGERTIP_PAIRS["right"][f][1] for f in ("thumb", "index", "middle", "ring", "pinky")]},
}
TIP_FINGERS = {"dex3": ("thumb", "index", "middle"),
               "brainco": ("thumb", "index", "middle", "ring", "pinky")}
CAM_KEYS = (("front_cam", "front"), ("ego_cam", "ego_view"))
TASK_TEXT = "hand the object from the right hand to the left hand and place it on the left"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def quat_look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    u0 = np.asarray(up, float)
    if abs(float(np.dot(f, u0))) > 0.999:
        u0 = np.array([1.0, 0.0, 0.0])
    left = np.cross(u0, f); left /= np.linalg.norm(left)
    R = np.column_stack([f, left, np.cross(f, left)])
    tr = float(np.trace(R))
    if tr > 0:
        sq = math.sqrt(tr + 1.0) * 2
        return (0.25 * sq, (R[2, 1] - R[1, 2]) / sq,
                (R[0, 2] - R[2, 0]) / sq, (R[1, 0] - R[0, 1]) / sq)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        sq = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / sq, 0.25 * sq,
                (R[0, 1] + R[1, 0]) / sq, (R[0, 2] + R[2, 0]) / sq)
    if R[1, 1] > R[2, 2]:
        sq = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / sq, (R[0, 1] + R[1, 0]) / sq,
                0.25 * sq, (R[1, 2] + R[2, 1]) / sq)
    sq = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / sq, (R[0, 2] + R[2, 0]) / sq,
            (R[1, 2] + R[2, 1]) / sq, 0.25 * sq)


def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
def object_spawn(kind, mass=0.05, friction=(1.0, 1.0, 0.0)):
    """steering_wheel = the shipped USD; cube / cylinder = primitive shapes.

    `mass` (kg) and `friction` = (static, dynamic, restitution). A 0.05 kg object in a
    loose three-finger hook flies off under small accelerations, so both are
    exposed on the CLI (--obj-mass / --obj-friction).
    """
    fs, fd, fr = friction
    if kind == "steering_wheel":
        return UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/"
                     f"pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=mass))
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=fs,
                                                        dynamic_friction=fd,
                                                        restitution=fr),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.1)))
    if kind == "cube":
        return sim_utils.CuboidCfg(size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE), **common)
    if kind == "bar":                        # lying horizontally, axis along world x
        if BAR_SHAPE == "box":               # square section: cannot roll on the risers
            sz = (2 * BAR_RADIUS, BAR_LENGTH, 2 * BAR_RADIUS) if BAR_AXIS == "y" else (BAR_LENGTH, 2 * BAR_RADIUS, 2 * BAR_RADIUS)
            return sim_utils.CuboidCfg(size=sz, **common)
        return sim_utils.CylinderCfg(radius=BAR_RADIUS, height=BAR_LENGTH, axis=("Y" if BAR_AXIS == "y" else "X"), **common)
    if STICK is not None:                    # upright square stick: grasped from above, cannot roll
        if len(STICK) > 2 and STICK[2]:      # held-out variant: ROUND stick of the same width
            return sim_utils.CylinderCfg(radius=STICK[0] / 2, height=STICK[1], axis="Z", **common)
        return sim_utils.CuboidCfg(size=(STICK[0], STICK[0], STICK[1]), **common)
    return sim_utils.CylinderCfg(radius=CYL_RADIUS, height=CYL_HEIGHT, axis="Z", **common)


def set_stick_com(scene, args):
    """Weighted-base stick: move the centre of mass down the stick axis (body frame) after sim.reset().

    A uniform stick gripped below its centre of mass balances in the hand and tips 15-20 deg in about
    half the episodes (demo batch A1). With the COM near the base every grip point is above it, the
    stick hangs straight in either hand, and it stands on the table up to atan(w/2 / com_height)."""
    if not getattr(args, "stick", 0) or getattr(args, "stick_com", None) is None:
        return None
    obj = scene["obj_cylinder"]
    view = obj.root_physx_view
    coms = view.get_coms().clone()
    coms[:, 0] = 0.0; coms[:, 1] = 0.0; coms[:, 2] = float(args.stick_com)
    idx = torch.arange(coms.shape[0], dtype=torch.int32, device=coms.device)
    view.set_coms(coms, idx)
    got = view.get_coms()[0].tolist()
    print(f"HANDOVER_COM stick com offset={args.stick_com} readback={[round(x, 4) for x in got[:3]]}", flush=True)
    return got


def table_top_z():
    """The measured packing-table surface height, or the hardcoded fallback."""
    return float(TABLE_TOP) if TABLE_TOP is not None else float(TABLE_TOP_Z)


def rest_z_for(kind, top=None):
    """World z of the object CENTRE when it sits at rest on the table."""
    return (table_top_z() if top is None else float(top)) + OBJ_HALF_Z[kind]


def riser_h_for(kind):
    """Height of the static risers the object starts on (bar only)."""
    return RISER_H if kind == "bar" else 0.0


def spawn_z(kind, top=None, clearance=0.005):
    """Spawn height = resting height (+ the riser it starts on) + a small clearance."""
    return rest_z_for(kind, top) + riser_h_for(kind) + clearance


def measure_table_top(scene, settle_fn, device, probe_xy=(0.0, 0.62), n_probe=100,
                      table_prim="/World/envs/env_0/PackingTable",
                      obj_prims=None, tag="HANDOVER_TABLE_TOP"):
    """Measure the real table top at runtime instead of trusting TABLE_TOP_Z.

    1. UsdGeom.BBoxCache world bound of the table prim (and of each parked object, which
       gives every object's true half height at the spawn scale).
    2. A physics probe: drop the cube on a clear part of the table and read where it
       settles.  The USD bbox of the packing table includes render-only structure well
       above the work surface (max z 1.083 while objects rest at 0.994), so the probe is
       authoritative and the bbox is the fallback / sanity bound.

    Sets the module globals TABLE_TOP and OBJ_HALF_Z and prints HANDOVER_TABLE_TOP.
    """
    global TABLE_TOP
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def bbox(path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.array(r.GetMin(), float), np.array(r.GetMax(), float)

    tb = bbox(table_prim)
    top_bbox = float(tb[1][2]) if tb is not None else float(TABLE_TOP_Z)
    if tb is not None:
        print(f"HANDOVER_TABLE_BBOX {table_prim} min={tb[0].round(4).tolist()} "
              f"max={tb[1].round(4).tolist()}", flush=True)
    obj_prims = obj_prims or {"steering_wheel": "/World/envs/env_0/ObjSteeringWheel",
                              "cube": "/World/envs/env_0/ObjCube",
                              "cylinder": "/World/envs/env_0/ObjCylinder",
                              "bar": "/World/envs/env_0/ObjBar"}
    for kind, path in obj_prims.items():
        b = bbox(path)
        if b is not None:
            h = float(b[1][2] - b[0][2]) / 2.0
            if 0.001 < h < 0.5:
                OBJ_HALF_Z[kind] = round(h, 4)
            print(f"HANDOVER_OBJ_BBOX {kind} size={(b[1] - b[0]).round(4).tolist()} "
                  f"half_z={OBJ_HALF_Z[kind]}", flush=True)

    top_probe = None
    try:
        place_objects(scene, "cube", probe_xy, top_bbox + 0.10, device)
        settle_fn(n_probe)
        zc = float(scene["obj_cube"].data.root_pos_w[0, 2])
        top_probe = zc - OBJ_HALF_Z["cube"]
        place_objects(scene, None, probe_xy, 0.3, device)       # park everything again
        settle_fn(5)
    except Exception as e:                                       # noqa: BLE001
        print(f"{tag} PROBE_FAILED ({type(e).__name__}: {e})", flush=True)

    top = top_bbox
    if top_probe is not None and 0.5 < top_probe < top_bbox + 0.01:
        top = top_probe
    TABLE_TOP = float(top)
    print(f"{tag} {TABLE_TOP:.4f} bbox_max={top_bbox:.4f} "
          f"probe={None if top_probe is None else round(top_probe, 4)} "
          f"hardcoded_was={TABLE_TOP_Z} rest_z="
          f"{ {k: round(rest_z_for(k), 4) for k in OBJ_HALF_Z} }", flush=True)
    return TABLE_TOP


def robot_cfg(robot):
    if robot == "dex3":
        return G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    if not os.path.exists(BRAINCO_USD):
        raise SystemExit(f"BrainCo USD missing: {BRAINCO_USD} (run scripts/convert_brainco_urdf.py)")
    cfg = G1_BRAINCO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.spawn.usd_path = BRAINCO_USD
    return cfg


def make_scene_cfg(args):
    """InteractiveSceneCfg with the table, marker, three parked objects and the robot."""
    place = args.place_xy
    tz = args.table_z
    # Static risers: only under the bar, parked off-table for every other object.  Their z
    # uses TABLE_TOP_PRIOR because the real top is only measured once the sim is running;
    # measure_table_top() prints the measured value so the two can be compared.
    global BAR_AXIS, RISER_SIZE, BAR_SHAPE, RISER_DX, STICK, THUMB_LEAD, BC_OPPOSE
    THUMB_LEAD = float(getattr(args, "thumb_lead", 0.0))
    BC_OPPOSE = tuple(args.bc_oppose) if getattr(args, "bc_oppose", None) else None
    STICK = (float(args.stick_w), float(args.stick_h), int(getattr(args, "stick_round", 0))) if getattr(args, "stick", 0) else None
    BAR_AXIS = getattr(args, "bar_axis", "x")
    BAR_SHAPE = getattr(args, "bar_shape", "cyl")
    RISER_DX = 0.11 if BAR_AXIS == "y" else 0.135   # 4 cm end overlap so +-2 cm pose noise keeps both ends seated
    RISER_SIZE = (0.14, 0.06, RISER_H) if BAR_AXIS == "y" else (0.03, 0.06, RISER_H)  # wide across the bar so pose noise keeps it seated
    from handover_common_args import resolve_obj_start as _ros0
    _ox, _oy = _ros0(args)
    _rz = TABLE_TOP_PRIOR + RISER_H / 2.0
    riser_xy = (([[_ox, _oy - RISER_DX, _rz], [_ox, _oy + RISER_DX, _rz]] if BAR_AXIS == "y"
                 else [[_ox - RISER_DX, _oy, _rz], [_ox + RISER_DX, _oy, _rz]])
                if args.object == "bar" else [[5.0, 3.0, 0.3], [5.4, 3.0, 0.3]])

    @configclass
    class HandoverSceneCfg(InteractiveSceneCfg):
        packing_table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/PackingTable",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0],
                                                    rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
        )
        obj_steering_wheel = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjSteeringWheel",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[*PARK_XY["steering_wheel"], 0.3], rot=[1, 0, 0, 0]),
            spawn=object_spawn("steering_wheel", mass=getattr(args, "obj_mass", 0.05),
                                 friction=tuple(getattr(args, "obj_friction",
                                                        (1.0, 1.0, 0.0)))))
        obj_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjCube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[*PARK_XY["cube"], 0.3], rot=[1, 0, 0, 0]),
            spawn=object_spawn("cube", mass=getattr(args, "obj_mass", 0.05),
                                 friction=tuple(getattr(args, "obj_friction",
                                                        (1.0, 1.0, 0.0)))))
        obj_cylinder = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjCylinder",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[*PARK_XY["cylinder"], 0.3], rot=[1, 0, 0, 0]),
            spawn=object_spawn("cylinder", mass=getattr(args, "obj_mass", 0.05),
                                 friction=tuple(getattr(args, "obj_friction",
                                                        (1.0, 1.0, 0.0)))))
        obj_bar = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjBar",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[*PARK_XY["bar"], 0.3], rot=[1, 0, 0, 0]),
            spawn=object_spawn("bar", mass=getattr(args, "obj_mass", 0.05),
                                 friction=tuple(getattr(args, "obj_friction",
                                                        (1.0, 1.0, 0.0)))))
        riser_l = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RiserL",
            init_state=AssetBaseCfg.InitialStateCfg(pos=riser_xy[0]),
            spawn=sim_utils.CuboidCfg(
                size=RISER_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25))))
        riser_r = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RiserR",
            init_state=AssetBaseCfg.InitialStateCfg(pos=riser_xy[1]),
            spawn=sim_utils.CuboidCfg(
                size=RISER_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25))))
        target_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TargetMarker",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[place[0], place[1], tz + 0.002]),
            spawn=sim_utils.CylinderCfg(
                radius=0.06, height=0.004, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2))),
        )
        robot = robot_cfg(args.robot)
        ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light",
                             spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),
                                                          intensity=3000.0))

    scene_cfg = HandoverSceneCfg(num_envs=1, env_spacing=3.0)
    scene_cfg.robot.init_state.pos = ROBOT_POS
    scene_cfg.robot.init_state.rot = ROBOT_ROT
    scene_cfg.robot.spawn.articulation_props.fix_root_link = True   # free-floating G1 collapses
    joint_pos = {
        ".*_shoulder_pitch_joint": 0.0, ".*_shoulder_roll_joint": 0.0,
        ".*_shoulder_yaw_joint": 0.0, ".*_elbow_joint": 0.0,
        ".*_wrist_yaw_joint": 0.0, ".*_wrist_roll_joint": 0.0, ".*_wrist_pitch_joint": 0.0,
        "waist_.*": 0.0, ".*_hip_.*": 0.0, ".*_knee_.*": 0.0, ".*_ankle_.*": 0.0,
        ".*_thumb_.*": 0.0, ".*_index_.*": 0.0, ".*_middle_.*": 0.0,
    }
    if args.robot == "brainco":
        joint_pos.update({".*_ring_.*": 0.0, ".*_pinky_.*": 0.0})
    scene_cfg.robot.init_state.joint_pos = joint_pos
    from handover_common_args import resolve_obj_start as _ros
    ox, oy = _ros(args)
    getattr(scene_cfg, f"obj_{args.object}").init_state.pos = [ox, oy, spawn_z(args.object)]
    ch = float(getattr(args, "stick_collar", 0.0) or 0.0)
    if getattr(args, "stick", 0) and ch > 0:  # static collar: 4 kinematic walls, 3 mm clearance per side
        g, t = args.stick_w / 2 + 0.003, 0.01
        walls = {"CollarXp": ((t, 2 * g + 2 * t, ch), (ox + g + t / 2, oy)),
                 "CollarXn": ((t, 2 * g + 2 * t, ch), (ox - g - t / 2, oy)),
                 "CollarYp": ((2 * g, t, ch), (ox, oy + g + t / 2)),
                 "CollarYn": ((2 * g, t, ch), (ox, oy - g - t / 2))}
        for nm, (size, (wx, wy)) in walls.items():
            setattr(scene_cfg, nm.lower(), AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + nm,
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, tz + ch / 2)),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.2, 0.2)))))
    if not args.no_video:
        attach_cameras(scene_cfg, args)
    return scene_cfg


def dump_hand_prims(prim_path="/World/envs/env_0/Robot", side="right", tag="HANDOVER_HANDPRIM",
                    limit=60):
    """Print every USD prim under one hand of the robot, with type and physics APIs.

    stage.Traverse() does NOT descend into instanced references (Isaac robot USDs are
    instanceable), which is why the old "_hand_" + CollisionAPI selector matched nothing.
    This walks instance proxies too, purely as a diagnostic.
    """
    try:
        import omni.usd
        from pxr import Usd
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            print(f"{tag} {prim_path} INVALID", flush=True)
            return 0
        it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))
        n = 0
        for prim in it:
            path = str(prim.GetPath())
            if f"{side}_hand" not in path:
                continue
            apis = [a for a in ("CollisionAPI", "MeshCollisionAPI", "MassAPI",
                                "RigidBodyAPI") if prim.HasAPI(getattr(UsdPhysics, a, object))]
            print(f"{tag} {path} type={prim.GetTypeName()} instance_proxy={prim.IsInstanceProxy()} "
                  f"physics={apis}", flush=True)
            n += 1
            if n >= limit:
                print(f"{tag} ... truncated at {limit}", flush=True)
                break
        print(f"{tag}_COUNT {n} prims under {side} hand", flush=True)
        return n
    except Exception as e:                                       # noqa: BLE001
        print(f"{tag} FAILED ({type(e).__name__}: {e})", flush=True)
        return 0


def bind_hand_material(robot, static=1.2, dynamic=1.1, restitution=0.0,
                       match="hand", tag="HANDOVER_MAT", dump_side="right"):
    """Set the PhysX material of every collision shape belonging to a hand link.

    A USD MaterialBindingAPI walk binds 0 prims here: the G1 USD is
    instanceable, so stage.Traverse() never reaches the collision meshes, and an
    instance proxy cannot be edited anyway.  The physics-view API used by IsaacLab's own
    randomize_rigid_body_material event writes the material straight into PhysX, per
    collision SHAPE, which is what actually decides hand/object friction.

    Returns the number of collision shapes written (HANDOVER_MAT count).
    """
    if dump_side:
        dump_hand_prims(side=dump_side)
    try:
        view = robot.root_physx_view
        body_names = list(robot.data.body_names)
        link_paths = list(view.link_paths[0])
        sim_view = getattr(robot, "_physics_sim_view", None)
        n_shapes = []
        if sim_view is not None:
            for lp in link_paths:
                n_shapes.append(int(sim_view.create_rigid_body_view(lp).max_shapes))
        if not n_shapes or sum(n_shapes) != int(view.max_shapes):
            per = int(view.max_shapes) // max(len(link_paths), 1)
            n_shapes = [per] * len(link_paths)
            print(f"{tag} shape-per-body parse fell back to {per} "
                  f"(total={view.max_shapes}, bodies={len(link_paths)})", flush=True)
        mats = view.get_material_properties().clone()
        n_bound, bound_bodies, before = 0, [], None
        for bi, name in enumerate(body_names):
            if match not in name:
                continue
            s0 = sum(n_shapes[:bi]); s1 = s0 + n_shapes[bi]
            if s1 <= s0 or s1 > mats.shape[1]:
                continue
            if before is None:
                before = mats[0, s0].tolist()
            if static is None:                      # measure only, leave PhysX untouched
                n_bound += s1 - s0
                bound_bodies.append(name)
                continue
            mats[:, s0:s1, 0] = static
            mats[:, s0:s1, 1] = dynamic
            mats[:, s0:s1, 2] = restitution
            n_bound += s1 - s0
            bound_bodies.append(name)
        print(f"{tag}_BEFORE existing hand material (static, dynamic, restitution) = {before}",
              flush=True)
        if n_bound and static is not None:
            view.set_material_properties(mats, torch.arange(mats.shape[0], dtype=torch.int32))
        print(f"{tag} bound static={static} dynamic={dynamic} restitution={restitution} "
              f"to {n_bound} hand collision prims over {len(bound_bodies)} links "
              f"{bound_bodies}", flush=True)
        return n_bound
    except Exception as e:                                   # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"{tag} FAILED ({type(e).__name__}: {e}) - object-side friction only",
              flush=True)
        return 0


def attach_cameras(scene_cfg, args):
    """ego_view + front rig, identical for the demo and rollout scripts."""
    lens = dict(focal_length=18.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.05, 20.0))
    w, h = args.cam_res
    for name, eye, tgt in (("front_cam", args.front_eye, args.front_target),
                           ("ego_cam", args.ego_eye, args.ego_target)):
        setattr(scene_cfg, name, TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=quat_look_at(eye, tgt),
                                            convention="world"),
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(**lens),
            width=w, height=h))


def place_objects(scene, active_kind, xy, z, device):
    """Put the active object at (xy, z) and park the other two far away."""
    for kind in OBJECT_KINDS:
        o = scene[f"obj_{kind}"]
        if kind == active_kind:
            pose = [xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0]
        else:
            pose = [PARK_XY[kind][0], PARK_XY[kind][1], 0.3, 1.0, 0.0, 0.0, 0.0]
        o.write_root_pose_to_sim(torch.tensor([pose], device=device))
        o.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        o.reset()


def grab_frames(scene, buf):
    for key, tag in CAM_KEYS:
        rgb = scene[key].data.output["rgb"]
        buf[tag].append(rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8))


def write_videos(buf, out_dir, stem, fps):
    import imageio.v2 as imageio
    paths = {}
    for _, tag in CAM_KEYS:
        d = os.path.join(out_dir, "cam", tag)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{stem}.mp4")
        with imageio.get_writer(p, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1) as wr:
            for fr in buf[tag]:
                wr.append_data(fr)
        paths[tag] = p
    return paths


# ---------------------------------------------------------------------------
# arm IK
# ---------------------------------------------------------------------------
class ArmIK:
    """One DifferentialIKController on the 7 arm joints of one side, world frame."""

    def __init__(self, robot, side, device, ik_lambda=0.05):
        names = list(robot.data.joint_names)
        self.side = side
        self.joint_ids = [names.index(f"{side}_{j}_joint") for j in ARM_JOINTS]
        self.body_name = f"{side}_wrist_yaw_link"
        self.body_id = list(robot.data.body_names).index(self.body_name)
        jac = robot.root_physx_view.get_jacobians()
        n_bodies = len(robot.data.body_names)
        if jac.shape[1] == n_bodies - 1:      # fixed base: root omitted, no base columns
            self.jac_body = self.body_id - 1
            self.jac_cols = self.joint_ids
        else:
            self.jac_body = self.body_id
            self.jac_cols = [i + 6 for i in self.joint_ids]
        self.ctrl = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                        ik_method="dls", ik_params={"lambda_val": ik_lambda}),
            num_envs=1, device=device)
        self.robot = robot
        self.device = device
        lim = robot.data.soft_joint_pos_limits[0, self.joint_ids]
        self.lo, self.hi = lim[:, 0], lim[:, 1]

    def wrist_pose_w(self):
        return self.robot.data.body_pose_w[:, self.body_id]

    def solve(self, pos_w, quat_w):
        cmd = torch.cat([pos_w, quat_w], dim=-1)
        self.ctrl.set_command(cmd)
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_body, :, self.jac_cols]
        ee = self.wrist_pose_w()
        q = self.robot.data.joint_pos[:, self.joint_ids]
        q_des = self.ctrl.compute(ee[:, 0:3], ee[:, 3:7], jac, q)
        return torch.clamp(q_des, self.lo, self.hi)


# ---------------------------------------------------------------------------
# hand model: index bookkeeping + closure targets for either robot
# ---------------------------------------------------------------------------
class HandModel:
    """Everything that depends on which hand is mounted.

    Attributes
        robot_kind        "dex3" | "brainco"
        arm_idx           14 articulation indices, robot order (identical on both robots)
        hand_idx          recorded hand block indices: dex3 14 (robot order) or brainco 12
                          (BRAINCO_ACTUATED order, resolved by name)
        hand_names        joint names of hand_idx
        all_hand_idx      every drivable hand joint (dex3 14, brainco 22) for limit checks
        side_idx[side]    subset of hand_idx for one side
        tip_ids[side]     body ids of the fingertip links
        dex3_hand_order   for brainco: the Dex3 column order the retargeting layer assumes
    """

    def __init__(self, robot, robot_kind, grip, dex3_hand_order=None):
        self.robot_kind = robot_kind
        self.grip = grip
        names = list(robot.data.joint_names)
        bodies = list(robot.data.body_names)
        self.names = names
        self.arm_idx = [i for i, n in enumerate(names)
                        if any(t in n for t in ("shoulder", "elbow", "wrist"))]
        self.lim = robot.data.soft_joint_pos_limits[0].cpu().numpy()          # closure targets (buffered)
        try:                                                                   # violations: HARD (URDF) limits
            self.hard_lim = robot.data.joint_pos_limits[0].cpu().numpy()
        except AttributeError:
            self.hard_lim = robot.data.default_joint_pos_limits[0].cpu().numpy()
        if robot_kind == "dex3":
            self.hand_idx = [i for i, n in enumerate(names)
                             if "_hand_" in n and any(r in n for r in DEX3_ROLES)]
            self.all_hand_idx = list(self.hand_idx)
            # closed target = grip x the soft-limit end with more travel from zero
            # open target = 0 clamped INTO the soft limits (index/middle_0 exclude 0, which
            # otherwise counts as a joint-limit violation on every step)
            self.close, self.open = {}, {}
            for i in self.hand_idx:
                lo, hi = float(self.lim[i, 0]), float(self.lim[i, 1])
                self.close[i] = hi * grip if abs(hi) >= abs(lo) else lo * grip
                self.open[i] = float(np.clip(0.0, lo, hi))
            self.dex3_hand_order = [names[i] for i in self.hand_idx]
            self.mimic_pairs = []
        else:
            missing = [n for n in BRAINCO_ACTUATED if n not in names]
            if missing:
                raise SystemExit(f"BrainCo USD lacks actuated joints {missing}; names={names}")
            self.hand_idx = [names.index(n) for n in BRAINCO_ACTUATED]
            self.all_hand_idx = self.hand_idx + [names.index(d) for d in BRAINCO_MIMIC if d in names]
            self.dex3_hand_order = list(dex3_hand_order or DEX3_HAND_ORDER)
            # Dex3 closure vector (the scripted Dex3 closed-hand target), in
            # dex3_hand_order, so that the BrainCo closed pose is literally its retarget.
            self.dex3_closed_vec = np.array([dex3_closed(n, grip) for n in self.dex3_hand_order])
            self.dex3_side_mask = {s: np.array([n.startswith(s) for n in self.dex3_hand_order])
                                   for s in ("left", "right")}
            # distal (articulation index) <- proximal (articulation index), ratio
            # (distal idx, proximal idx, ratio, hard upper) -- same rule as apply_mimic
            self.mimic_pairs = [(names.index(d), names.index(p), r,
                                 0.9 * BRAINCO_UPPER[d.split("_", 1)[1].rsplit("_joint", 1)[0]])  # soft limit
                                for d, (p, r) in BRAINCO_MIMIC.items() if d in names and p in names]
            self.close = None
        self.hand_names = [names[i] for i in self.hand_idx]
        self.side_idx = {s: [i for i in self.hand_idx if names[i].startswith(s)]
                         for s in ("left", "right")}
        self.tip_ids, self.tip_names = {}, {}
        for s in ("left", "right"):
            ids = []
            for b in TIP_BODIES[robot_kind][s]:
                if b not in bodies:
                    raise SystemExit(f"fingertip body {b} not in articulation bodies: {bodies}")
                ids.append(bodies.index(b))
            self.tip_ids[s] = ids
            self.tip_names[s] = list(TIP_BODIES[robot_kind][s])
        self.n_state = N_ARM + len(self.hand_idx)
        if len(self.arm_idx) != N_ARM or self.n_state != STATE_DIM[robot_kind]:
            raise SystemExit(f"{robot_kind}: expected {STATE_DIM[robot_kind]}-D state, got "
                             f"{len(self.arm_idx)} arm + {len(self.hand_idx)} hand")

    # ---- closure -> hand block (scripted demos) ----
    def hand_block_from_grip(self, grip_R, grip_L):
        """Hand block (dex3 14 / brainco 12) for scalar closure fractions per side."""
        if self.robot_kind == "dex3":
            return np.array([lerp(self.open[i], self.close[i],
                                  grip_L if self.names[i].startswith("left") else grip_R)
                             for i in self.hand_idx], dtype=np.float64)
        if BC_OPPOSE is not None:
            # BrainCo-native grasp (not a Dex3 retarget): the thumb is swung fully across the palm
            # (metacarpal m_pre) from the start, so its flexion sweeps horizontally toward the curling
            # fingers and the hand forms a C around an upright stick. Fingers close over the first 70%
            # of the ramp, the thumb flexes over the last 70%. Order = BRAINCO_ACTUATED per side:
            # thumb_metacarpal, thumb_proximal, index, middle, ring, pinky.
            m_pre, p_close, f_close = BC_OPPOSE[:3]
            # optional ramp windows (fractions of the close ramp): thumb [t0, t1], fingers [f0, f1]
            t0, t1, f0, f1 = BC_OPPOSE[3:7] if len(BC_OPPOSE) >= 7 else (0.3, 1.0, 0.0, 0.7)
            rp = BC_OPPOSE[7] if len(BC_OPPOSE) >= 8 else 1.0   # ring/pinky closure scale (0 = thumb+index+middle only)
            out = []
            for sd, g in (("L", float(grip_L)), ("R", float(grip_R))):
                gf = min(max((g - f0) / (f1 - f0), 0.0), 1.0); gt = min(max((g - t0) / (t1 - t0), 0.0), 1.0)
                out += [m_pre * BC_SWING[sd], gt * p_close, gf * f_close, gf * f_close,
                        gf * f_close * rp, gf * f_close * rp]
            return np.asarray(out, dtype=np.float64)
        g = np.where(self.dex3_side_mask["left"], grip_L, grip_R).astype(np.float64)
        if THUMB_LEAD > 0:      # BrainCo: thumb closes first (blocks the stick's escape), then the fingers
            tm = np.array(["thumb" in n for n in self.dex3_hand_order])
            g = np.where(tm, np.clip(g / THUMB_LEAD, 0.0, 1.0),
                         np.clip((g - THUMB_LEAD) / (1.0 - THUMB_LEAD), 0.0, 1.0))
        v = self.dex3_closed_vec * g
        return self.hand_block_from_dex3(v)

    def hand_block_from_dex3(self, dex3_hand_14):
        """Dex3 14-D hand command -> this robot's hand block (identity on dex3)."""
        if self.robot_kind == "dex3":
            return np.asarray(dex3_hand_14, dtype=np.float64)
        return retarget_hand(np.asarray(dex3_hand_14, dtype=np.float64),
                             self.dex3_hand_order, self.grip)

    def hand_block_to_dex3(self, hand_block):
        """Inverse: this robot's hand block -> Dex3 14-D (identity on dex3)."""
        if self.robot_kind == "dex3":
            return np.asarray(hand_block, dtype=np.float64)
        return inverse_retarget_hand(np.asarray(hand_block, dtype=np.float64),
                                     self.dex3_hand_order, self.grip)

    # ---- writing ----
    def write_hand(self, targets, hand_block):
        """Write a hand block into the (1, n_joints) target tensor; fills mimic distals."""
        t = torch.as_tensor(np.asarray(hand_block, dtype=np.float32), device=targets.device)
        targets[0, self.hand_idx] = t
        for d, p, r, hi in self.mimic_pairs:
            targets[0, d] = torch.clamp(targets[0, p] * r, min=0.0, max=float(hi))
        return targets

    def write_arms(self, targets, arm14):
        targets[0, self.arm_idx] = torch.as_tensor(np.asarray(arm14, dtype=np.float32),
                                                   device=targets.device)
        return targets

    # ---- reading ----
    def state(self, q):
        return np.concatenate([q[self.arm_idx], q[self.hand_idx]]).astype(np.float32)

    def action_vec(self, targets_np):
        return np.concatenate([targets_np[self.arm_idx], targets_np[self.hand_idx]]).astype(np.float32)

    def limit_violation(self, q, tol=0.02):
        hq = q[self.all_hand_idx]
        lim = self.hard_lim           # a violation means past the joint's real limit, not Isaac's soft buffer
        bad = (hq > lim[self.all_hand_idx, 1] + tol) | (hq < lim[self.all_hand_idx, 0] - tol)
        if not hasattr(self, "jl_counts"):
            self.jl_counts = np.zeros(len(self.all_hand_idx), dtype=int)
        self.jl_counts += bad.astype(int)
        return bool(np.any(bad))

    def jl_report(self):
        """Per-joint violation counts (name, count, soft lo/hi) for joints that ever violated; resets."""
        if not hasattr(self, "jl_counts"):
            return []
        out = [(self.names[j], int(c), float(self.hard_lim[j, 0]), float(self.hard_lim[j, 1]))
               for j, c in zip(self.all_hand_idx, self.jl_counts) if c > 0]
        self.jl_counts[:] = 0
        return out

    def tip_row_names(self):
        """Row labels of ep{i}_tips.npy: 2 wrists then left tips then right tips."""
        return (["left_wrist_yaw_link", "right_wrist_yaw_link"]
                + self.tip_names["left"] + self.tip_names["right"])


def mimic_ok(hand):
    return hand.robot_kind == "dex3" or len(hand.mimic_pairs) == 10


# ---------------------------------------------------------------------------
# reference fingertips (M2) loaded from a Dex3 run
# ---------------------------------------------------------------------------
class TipReference:
    """ep{i}_tips.npy files of a Dex3 run, for palm-relative fingertip tracking error."""

    def __init__(self, ref_dir):
        self.dir = ref_dir
        summ = os.path.join(ref_dir, "handover_summary.json")
        self.rows = None
        if os.path.exists(summ):
            with open(summ) as f:
                d = json.load(f)
            self.rows = d.get("tip_rows")
            self.robot = d.get("robot", "dex3")
        if self.rows is None:
            self.robot = "dex3"
            self.rows = (["left_wrist_yaw_link", "right_wrist_yaw_link"]
                         + TIP_BODIES["dex3"]["left"] + TIP_BODIES["dex3"]["right"])

    def load(self, ep):
        p = os.path.join(self.dir, f"ep{ep}_tips.npy")
        if not os.path.exists(p):           # policy rollouts may run more episodes than the reference
            files = sorted(glob.glob(os.path.join(self.dir, "ep*_tips.npy")))
            if not files:
                return None
            p = files[ep % len(files)]
        return np.load(p)

    def wrist_error(self, tips, ep):
        """Mean world-frame wrist position error (m) vs the reference, both sides, overlapping steps."""
        ref = self.load(ep)
        if ref is None or len(tips) == 0:
            return None
        T = min(len(ref), len(tips))
        e = [float(np.linalg.norm(tips[:T, k] - ref[:T, self.rows.index(n)], axis=-1).mean())
             for k, n in enumerate(("left_wrist_yaw_link", "right_wrist_yaw_link"))]
        return {"left": e[0], "right": e[1], "mean": float(np.mean(e)), "steps": int(T)}

    def error(self, tips, hand, ep):
        """Mean palm-relative fingertip error (m) over overlapping steps, primary fingers only.

        Ring/pinky (brainco) are compared against the Dex3 middle tip and reported
        separately as informational.
        """
        ref = self.load(ep)
        if ref is None or len(tips) == 0:
            return None
        T = min(len(ref), len(tips))
        mine = hand.tip_row_names()
        errs, info = [], []
        for s in ("left", "right"):
            rw = ref[:T, self.rows.index(f"{s}_wrist_yaw_link")]
            mw = tips[:T, mine.index(f"{s}_wrist_yaw_link")]
            for finger in TIP_FINGERS[hand.robot_kind]:
                src_finger = "middle" if finger in ("ring", "pinky") else finger
                rb = TIP_BODIES[self.robot][s][TIP_FINGERS[self.robot].index(src_finger)]
                mb = TIP_BODIES[hand.robot_kind][s][TIP_FINGERS[hand.robot_kind].index(finger)]
                e = np.linalg.norm((tips[:T, mine.index(mb)] - mw) - (ref[:T, self.rows.index(rb)] - rw), axis=-1)
                (info if finger in ("ring", "pinky") else errs).append(float(e.mean()))
        return {"mean_primary": float(np.mean(errs)), "per_finger_primary": errs,
                "mean_informational": float(np.mean(info)) if info else None, "steps": int(T)}


# ---------------------------------------------------------------------------
# per-episode metric tracker (shared by demo generation and policy evaluation)
# ---------------------------------------------------------------------------
class EpisodeTracker:
    """Staged handover metrics from object / wrist positions, one call per control step.

    Every stage is PHASE-GATED: it can only latch during or after the scripted phase
    that is supposed to produce it (grasp -> the right-hand close, transfer -> the
    left-hand close, place -> the lowering phase). Without that gate the stages latch
    on geometry that is already true before anything has happened -- e.g. with the home
    left wrist was 0.27 m from the object (so dL <= 0.35 held at step 1) and the
    untouched object at its spawn (0.05, 0.38) was 0.10 m from the place target
    (-0.05, 0.38), inside place_radius 0.12, so near_xy held at step 1 too.
    t_place additionally needs the object RESTING and both hands clear for
    place_hold consecutive steps, and success re-validates at the FINAL frame.
    """

    # milestone phase a stage may first fire in (first name found in phase_names wins)
    GATE_PHASES = {"grasp": ("r_close", "close_rp", "rclose"),
                   "transfer": ("l_close", "lclose"),
                   "place": ("lower",)}

    def __init__(self, z0, place_xy, root_z_init, vel_spike=0.6, lower_phase_index=None,
                 place_radius=0.08, phase_names=None, rest_z=None, rest_tol=REST_TOL,
                 hand_clear=HAND_CLEAR, place_hold=PLACE_HOLD, hold_radius=0.25,
                 release_radius=0.35):
        self.place_radius = float(place_radius)
        self.z0 = float(z0)
        self.rest_z = float(z0 if rest_z is None else rest_z)   # object centre at rest on the table
        self.rest_tol = float(rest_tol)
        self.hand_clear = float(hand_clear)
        self.place_hold = int(place_hold)
        self.hold_radius = float(hold_radius)
        self.release_radius = float(release_radius)
        self.place = np.asarray(place_xy, np.float64)[:2]
        self.root_z_init = float(root_z_init)
        self.vel_spike = vel_spike
        self.lower_phase = lower_phase_index
        self.phase_names = list(phase_names) if phase_names else None
        self.gates = {}
        for stage, cands in self.GATE_PHASES.items():
            idx = None
            if self.phase_names:
                for nm in cands:
                    if nm in self.phase_names:
                        idx = self.phase_names.index(nm)
                        break
            self.gates[stage] = idx
        self.t_grasp = self.t_transfer = self.t_place = None
        self.t_transfer_proxy = None            # old dL<dR proxy, logged only
        self.dropped = self.fell = False
        self.was_elev = False
        self.jl_viol = 0
        self.contact_spikes = 0
        self.prev_v = None
        self.step = 0
        self._place_run = 0                     # consecutive steps the place condition holds
        self.last_op = self.last_dR = self.last_dL = None

    def check_spawn(self, obj_p0, log_prefix=""):
        """Warn when the spawn already satisfies the place test (run-7 configuration)."""
        d = float(np.linalg.norm(np.asarray(obj_p0, np.float64)[:2] - self.place))
        if d <= self.place_radius:
            print(f"{log_prefix}HANDOVER_WARN spawn is already inside place_radius "
                  f"(d={d:.3f} <= {self.place_radius}): the place test is only meaningful "
                  f"because of the phase gate + hold + final re-validation", flush=True)
        return d

    # -- gating ------------------------------------------------------------
    def _allowed(self, stage, phase_index):
        """True if the current phase is at or after the stage's own phase.

        No phase information (policy rollout) -> ungated; the dwell and final-frame
        checks still apply.
        """
        gate = self.gates.get(stage)
        if gate is None or phase_index is None:
            return True
        return int(phase_index) >= gate

    def update(self, op, ov, pR, pL, root_z, jl_bad, phase_index=None, phase_name=None,
               log_prefix=""):
        self.step += 1
        op = np.asarray(op, np.float64)
        if phase_index is None and phase_name is not None and self.phase_names:
            phase_index = (self.phase_names.index(phase_name)
                           if phase_name in self.phase_names else None)
        dR, dL = float(np.linalg.norm(pR - op)), float(np.linalg.norm(pL - op))
        self.last_op, self.last_dR, self.last_dL = op.copy(), dR, dL
        elev = (op[2] - self.z0) >= 0.05
        if (self.t_grasp is None and elev and dR <= self.release_radius
                and self._allowed("grasp", phase_index)):
            self.t_grasp = self.step
        if self.t_grasp is not None and self.t_transfer_proxy is None and elev and dL < dR:
            self.t_transfer_proxy = self.step
        # real transfer: right hand has let go and moved away, object still up, left hand holds it
        if (self.t_grasp is not None and self.t_transfer is None and elev
                and dR > self.release_radius and dL <= self.hold_radius
                and self._allowed("transfer", phase_index)):
            self.t_transfer = self.step
        near_xy = float(np.linalg.norm(op[:2] - self.place)) <= self.place_radius
        resting = abs(op[2] - self.rest_z) < self.rest_tol
        placed_now = (near_xy and resting and dR > self.hand_clear and dL > self.hand_clear
                      and self.t_transfer is not None
                      and self._allowed("place", phase_index))
        self._place_run = self._place_run + 1 if placed_now else 0
        if self.t_place is None and self._place_run >= self.place_hold:
            self.t_place = self.step - self.place_hold + 1      # first step of the held window
            print(f"{log_prefix}HANDOVER_PLACE step={self.t_place} obj={op.round(3).tolist()} "
                  f"dR={dR:.3f} dL={dL:.3f}", flush=True)
        if (self.was_elev and not elev and self.t_place is None and not self.dropped
                and self.t_grasp is not None):   # a wheel riding up on the fingers during the
                                                 # bulldozing reach and settling back is not a drop
            # scripted: a descent before the lowering phase or away from the target is a
            # drop; policy (no phases): any descent away from the target is a drop
            early = (phase_index is not None and self.lower_phase is not None
                     and phase_index < self.lower_phase)
            if early or not near_xy:
                self.dropped = True
                print(f"{log_prefix}HANDOVER_DROP step={self.step} obj={op.round(3).tolist()}",
                      flush=True)
        self.was_elev = self.was_elev or elev
        if (self.root_z_init - float(root_z)) > 0.2:
            self.fell = True
        if jl_bad:
            self.jl_viol += 1
        v = np.asarray(ov, np.float64)
        if self.prev_v is not None and float(np.linalg.norm(v - self.prev_v)) > self.vel_spike:
            self.contact_spikes += 1
        self.prev_v = v
        return elev

    # -- final-frame re-validation ----------------------------------------
    @property
    def final_near_xy(self):
        if self.last_op is None:
            return False
        return bool(float(np.linalg.norm(self.last_op[:2] - self.place)) <= self.place_radius)

    @property
    def final_resting(self):
        if self.last_op is None:
            return False
        return bool(abs(float(self.last_op[2]) - self.rest_z) < self.rest_tol)

    def check_free(self, pos, quat_wxyz, points, half_len, half_w, tilt_band=(15.0, 75.0), clear=0.03):
        """Stick objects: is the placed stick standing or lying on its own at the last frame?

        A stick resting on a flat table with nothing else touching it is either upright or lying flat,
        so a final tilt inside `tilt_band` means something (a hand) is propping it up. `points` are
        the wrist and fingertip positions of both hands; any of them within `clear` of the stick
        surface means the hand has not let go. The centre-distance hand-clear test alone let a stick
        leaning on the fingers count as placed.
        """
        w, x, y, z = [float(v) for v in quat_wxyz]
        ax = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        ax = ax / max(np.linalg.norm(ax), 1e-9)
        tilt = math.degrees(math.acos(min(1.0, abs(float(ax[2])))))
        p = np.asarray(pos, np.float64)
        a, b = p - half_len * ax, p + half_len * ax
        def seg_d(q):
            t = float(np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0.0, 1.0))
            return float(np.linalg.norm(q - (a + t * (b - a))))
        d = min(seg_d(np.asarray(q, np.float64)) for q in points) - half_w
        self.final_tilt, self.final_hand_clear = tilt, d
        self.final_free = bool(not (tilt_band[0] < tilt < tilt_band[1]) and d > clear)
        return self.final_free

    @property
    def final_ok(self):
        """The latches are not enough: the object must STILL be placed at the last frame
        (and, for sticks when check_free() ran, standing or lying on its own, released)."""
        free = getattr(self, "final_free", None)
        return bool(self.final_near_xy and self.final_resting and (free is None or free))

    @property
    def success(self):
        return bool(self.t_place is not None and not self.dropped and not self.fell
                    and self.final_ok)

    def record(self):
        return {"t_grasp": self.t_grasp, "t_transfer": self.t_transfer,
                "transfer_proxy": self.t_transfer_proxy, "t_place": self.t_place,
                "dropped": self.dropped, "fell": self.fell,
                "joint_limit_violations": self.jl_viol, "contact_spikes": self.contact_spikes,
                "completion_step": self.t_place, "frames": self.step,
                "place_xy": self.place.round(4).tolist(), "place_radius": self.place_radius,
                "rest_z": round(self.rest_z, 4), "rest_tol": self.rest_tol,
                "place_hold": self.place_hold, "hand_clear": self.hand_clear,
                "phase_gates": {k: (self.phase_names[v] if (self.phase_names and v is not None)
                                    else None) for k, v in self.gates.items()},
                "final_near_xy": self.final_near_xy, "final_resting": self.final_resting,
                "final_dR": None if self.last_dR is None else round(self.last_dR, 4),
                "final_dL": None if self.last_dL is None else round(self.last_dL, 4),
                "final_ok": self.final_ok,
                "final_free": getattr(self, "final_free", None),
                "final_tilt_deg": getattr(self, "final_tilt", None),
                "final_hand_clear_m": getattr(self, "final_hand_clear", None),
                "success": bool(self.success)}

    def failure_category(self):
        if self.success:
            return "success"
        if self.fell:
            return "fall"
        if self.dropped:
            if self.t_transfer is None:
                return "drop_before_transfer"
            return "drop_after_transfer"
        if self.t_grasp is None:
            return "no_grasp"
        if self.t_transfer is None:
            return "no_transfer"
        if self.t_place is None:
            return "no_place"
        if getattr(self, "final_free", None) is False and self.final_near_xy and self.final_resting:
            return "not_released"
        return "moved_after_place"


ARM_MIRROR = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])   # ARM_JOINTS order


def mirror_quat_x(q):
    """Rotation reflected across the world x=0 plane, (w, x, y, z) -> (w, x, -y, -z)."""
    q = np.asarray(q, np.float64)
    return np.array([q[0], q[1], -q[2], -q[3]])


def print_world_bboxes(paths, tag="HANDOVER_BBOX"):
    """World-space axis-aligned bounds of USD prims (spawn pose, not the physx pose)."""
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"{tag} {path} INVALID", flush=True); continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        print(f"{tag} {path} min={lo.round(4).tolist()} max={hi.round(4).tolist()} "
              f"size={(hi - lo).round(4).tolist()}", flush=True)


def tips_now(robot, hand):
    """(2 + n_tips, 3) world positions: both wrists then left tips then right tips."""
    bp = robot.data.body_pos_w[0]
    ids = ([list(robot.data.body_names).index("left_wrist_yaw_link"),
            list(robot.data.body_names).index("right_wrist_yaw_link")]
           + hand.tip_ids["left"] + hand.tip_ids["right"])
    return bp[ids].cpu().numpy().astype(np.float32)


def load_dex3_hand_order(actions_dir):
    """Hand joint order recorded in a Dex3 run's handover_summary.json, if present."""
    p = os.path.join(actions_dir, "handover_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    names = d.get("joint_names")
    hidx = d.get("hand_joint_ids")
    if names and hidx:
        return [names[i] for i in hidx]
    return None
