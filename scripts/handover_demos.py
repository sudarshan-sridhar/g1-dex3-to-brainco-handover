r"""Scripted bimanual handover demonstrations, robot-agnostic (--robot dex3 | brainco).

Each arm is driven by a differential IK controller (pose, damped least squares) on its
7 arm joints, with {side}_wrist_yaw_link as the end-effector and targets in the world
frame. The hands follow a closure schedule. The task runs as a fixed sequence of phases
(reach, grasp, lift, carry, left-hand grasp, right release, place, release, retreat), and
every target is computed from the measured object pose at the start of its phase.

ROBOTS
    dex3     G1_29DOF_CFG            28-D = arms(14) | Dex3 hands(14, robot order)
    brainco  G1_BRAINCO_CFG          26-D = arms(14) | BrainCo actuated(12, BRAINCO_ACTUATED order)
             (assets_brainco/usd/g1_29dof_brainco.usd, distal joints driven by apply_mimic
             every step. By default the hand targets are the Dex3 closure mapped through the
             retargeting layer (Stage B); with --bc-oppose the BrainCo hands use their own
             opposition grasp, which is how the Stage C demonstrations are recorded)

MODES
    --mode scripted                 compute IK + closure schedule (Stage A on dex3, Stage B
                                    "scripted retarget" on brainco)
    --replay-actions <dir>          no IK: replay ep*_actions.npy recorded by a Dex3 run;
                                    columns 0:14 are the arm joint targets, columns 14:28 go
                                    through retarget_hand when --robot brainco (requirement
                                    1.4: Dex3 trajectories retargeted, no policy, no finetune).
                                    --episodes caps how many files are replayed; the object
                                    start pose is re-sampled with the SAME --seed/--pose-noise
                                    so ep{i} sees the jitter ep{i} saw when recorded.

OUTPUT  <out>/ep{i}_actions.npy, ep{i}_states.npy (T, 28|26), ep{i}_tips.npy (T, 2+n_tips, 3)
        [2 wrists, left tips, right tips; row names in handover_summary.json "tip_rows"],
        cam/{front,ego_view}/demo_%06d.mp4, handover_summary.json
"""

import argparse
import os

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--out", type=str, default=None,
                    help="default results/handover_<robot>")
parser.add_argument("--mode", type=str, default="scripted", choices=["scripted"])
parser.add_argument("--replay-actions", type=str, default=None,
                    help="dir with ep*_actions.npy (T, 28) from a Dex3 run; replays them")
parser.add_argument("--stage-label", type=str, default=None,
                    help="A|B|C tag in file names; default A for dex3, B for brainco")
parser.add_argument("--dry-run", action="store_true", default=False)
# phase step counts (each = --decimation physics steps)
parser.add_argument("--n-reach", type=int, default=40)
parser.add_argument("--n-descend", type=int, default=30)
parser.add_argument("--n-rclose", type=int, default=25)
parser.add_argument("--n-lift", type=int, default=30)
parser.add_argument("--n-carry", type=int, default=50)
parser.add_argument("--n-lreach", type=int, default=50)
parser.add_argument("--n-lclose", type=int, default=25)
parser.add_argument("--n-ropen", type=int, default=20)
parser.add_argument("--n-rretract", type=int, default=30)
parser.add_argument("--n-lcarry", type=int, default=50)
parser.add_argument("--n-lower", type=int, default=25)
parser.add_argument("--n-lopen", type=int, default=20)
parser.add_argument("--n-lretract", type=int, default=30)
parser.add_argument("--n-hold", type=int, default=40,
                    help="--l-replay: steps the right hand holds the wheel still at the midline "
                         "before the left hand sweeps in")
# geometry (world frame; robot faces +y, robot-right is +x)
parser.add_argument("--mid-xy", type=float, nargs=2, default=(0.0, 0.33))
parser.add_argument("--mid-height", type=float, default=0.15,
                    help="right wrist height above the wheel rest z at the handover (wheel hangs below)")
parser.add_argument("--roll-deg", type=float, default=90.0,
                    help="right wrist roll about world y after carry: wheel plane horizontal -> vertical")
parser.add_argument("--n-roll", type=int, default=40,
                    help="--handover roll: the wrist roll is a SEPARATE slow phase after the "
                         "carry, run once the object is stationary at the midline")
parser.add_argument("--handover", type=str, default="mirror", choices=["mirror", "roll", "bar", "stick"],
                    help="mirror = run-1 geometry (left target = x-mirror of the right grasp offset); "
                         "roll = wheel rolled vertical, left takes the bottom rim (experimental)")
parser.add_argument("--phases", type=str, default="full", choices=["full", "grasp_only"],
                    help="grasp_only = right-hand grasp test: reach, descend, slide_in, r_close, "
                         "lift, then hold still for --n-hold-still steps and report the "
                         "object height above its rest pose and its distance to the right wrist")
parser.add_argument("--n-hold-still", type=int, default=60)
parser.add_argument("--n-cal", type=int, default=25,
                    help="steps used to close/open both hands at the home pose to MEASURE "
                         "the grasp pocket (wrist -> closed-fingertip centroid)")
parser.add_argument("--bar-yaw-deg", type=float, default=90.0,
                    help="right wrist rotation about world +z before grasping the bar; +90 "
                         "turns the fingers to point in -x (along the bar, inwards) and puts "
                         "the closure axis along y, perpendicular to the bar, so the hand "
                         "descends onto the bar and pinches it between the thumb (-y side) "
                         "and index+middle (+y side).  The left wrist gets -this.")
parser.add_argument("--bar-roll-deg", type=float, default=0.0,
                    help="right wrist rotation about world +y before grasping the bar; +90 "
                         "puts the thumb ABOVE the bar and index+middle UNDER it (the left "
                         "wrist gets -this, so both hands approach from +y with the same grip)")
parser.add_argument("--bar-grip-x", type=float, default=0.08,
                    help="|x| offset from the bar centre where each hand grasps: the right "
                         "hand at +this, the left hand at -this")
parser.add_argument("--bar-pre", type=float, default=0.0,
                    help="stand-off in -y before sliding the open hand onto the bar")
parser.add_argument("--bar-above", type=float, default=0.08)
parser.add_argument("--bar-lift", type=float, default=0.15)
parser.add_argument("--wrist-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="manual world correction added to the IK wrist target at the bar")
parser.add_argument("--pocket", type=float, nargs=3, default=None,
                    help="override the MEASURED grasp pocket (world offset wrist->pocket at "
                         "the home wrist orientation)")
parser.add_argument("--obj-mass", type=float, default=None,
                    help="object mass (kg); 0.05 flies out of a loose hook, 0.25 does not")
parser.add_argument("--obj-friction", type=float, nargs=3, default=(1.0, 1.0, 0.0),
                    metavar=("STATIC", "DYNAMIC", "RESTITUTION"),
                    help="physics material on the object AND the hand collision meshes")
parser.add_argument("--phase-scale", type=float, default=1.0,
                    help="multiply every phase length (lower accelerations)")
parser.add_argument("--place-radius", type=float, default=0.12,
                    help="xy distance to the target marker that counts as placed")
parser.add_argument("--l-pre-offset", type=float, nargs=3, default=(-0.05, 0.0, -0.10),
                    help="--handover roll: left pre-grasp stand-off from the bottom rim "
                         "(default: outside in -x and BELOW, so the wrist comes up under it)")
parser.add_argument("--r-lift-away", type=float, nargs=3, default=(0.0, 0.0, 0.12),
                    help="--handover roll: how the right wrist leaves after r_open")
parser.add_argument("--l-grasp-offset", type=float, nargs=3, default=(-0.035, -0.078, -0.03),
                    help="left wrist - bottom rim point of the hanging wheel (palm +x pinches the tube)")
parser.add_argument("--lift-height", type=float, default=0.15)
parser.add_argument("--reach-height", type=float, default=0.15)
parser.add_argument("--grasp-offset", type=float, nargs=3, default=(0.055, -0.10, -0.02),
                    help="right wrist - wheel root at grasp: hand OUTSIDE the ring on the robot-right "
                         "side, palm inward, fingers (pointing +y) hook the +x rim from the side")
parser.add_argument("--pre-offset", type=float, default=0.06,
                    help="lateral stand-off (m) for descend / approach so the fingers clear the rim")
parser.add_argument("--n-slide", type=int, default=20)
parser.add_argument("--n-lpre", type=int, default=40)
parser.add_argument("--n-out", type=int, default=20)
parser.add_argument("--wrist-quat", type=str, default="hold", choices=["hold", "base"])
parser.add_argument("--ik-lambda", type=float, default=0.05)
# right-hand grasp taken from a proven Dex3 LEFT-hand replay (results/dex3_batch60), mirrored
parser.add_argument("--grasp-replay", type=str, default=None,
                    help="ep*_actions.npy (T, 28) of a successful Dex3 episode; the left arm "
                         "columns are mirrored onto the right arm for reach/close/lift. '' = pure IK")
parser.add_argument("--grasp-frames", type=int, nargs=3, default=(0, 130, 180),
                    help="replay frame at reach start, close start, close end")
parser.add_argument("--lift-frame", type=int, default=210, help="replay frame where the lift ends")
parser.add_argument("--grasp-stride", type=int, default=2, help="replay frames per control step")
parser.add_argument("--l-side-offset", type=float, default=0.0,
                    help="extra -x offset of the left grasp target at the midline (m)")
parser.add_argument("--obj-pose-exact", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="after the robot has settled, teleport the object to EXACTLY this world "
                         "pose with zero velocity. The wheel spawn overlaps the home right hand, "
                         "so the settle drifts it chaotically; this pins the pose the "
                         "open-loop mirrored replay was tuned against")
parser.add_argument("--hand-friction", type=float, nargs=3, default=(-1.0, -1.0, -1.0),
                    metavar=("STATIC", "DYNAMIC", "RESTITUTION"),
                    help="PhysX material written to every hand collision shape. DEFAULT IS OFF "
                         "('-1 -1 -1' = count the shapes and log the existing value, leave PhysX "
                         "untouched), because the robot asset already defines its hand material "
                         "(0.5/0.5/0.0) and changing it changes the grasp")
parser.add_argument("--l-y-offset", type=float, default=0.0, help="added to off_L y (0.258 = far rim instead of near rim)")
parser.add_argument("--pre-vec", type=float, nargs=3, default=None, help="left pre-grasp approach vector, overrides --pre-offset")
parser.add_argument("--l-receive", type=float, nargs=3, default=None, help="mirror mode: left wrist waits open at this world pose; the right carries the wheel into it")
parser.add_argument("--recv-approach", type=float, default=0.0, help="with --l-receive: stop the carry this far short in y, then slide the rim +y into the left hook")
parser.add_argument("--l-wait-after-lift", type=int, default=0, help="1: the left moves to --l-receive after the right has lifted, not before the grasp")
parser.add_argument("--l-yaw-deg", type=float, default=0.0, help="extra world-z yaw applied to the left grasp quat")
parser.add_argument("--l-quat-late", type=int, default=0, help="1: apply the left grasp quat at l_pre, not at carry_mid (keeps the left hand still during the carry)")
parser.add_argument("--bar-grip-far", type=float, default=0.09, help="bar handover: +y offset from the bar centre where the left hand grasps")
parser.add_argument("--stick-lo", type=float, default=0.05, help="stick: right grip point below the centre")
parser.add_argument("--stick-hi", type=float, default=0.06, help="stick: left grip point above the centre")
parser.add_argument("--l-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="stick: extra offset on the left grasp target (e.g. -x for more finger clearance)")
parser.add_argument("--r-close-advance", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="stick/bar: wrist moves this much WHILE the right hand closes (palm comes onto the object instead of the fingers dragging it)")
parser.add_argument("--l-close-advance", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="stick: same for the left hand")
parser.add_argument("--straighten-max", type=float, default=0.03, help="stick: max wrist move per straighten pass")
parser.add_argument("--lower-clear", type=float, default=0.005, help="stick: lowest point this far above the table at the end of lowering (negative = press)")
parser.add_argument("--bar-pitch-deg", type=float, default=0.0, help="stick/bar: wrist rotation about world x (-90 = fingers pointing down)")
parser.add_argument("--r-pre-vec", type=float, nargs=3, default=None, help="stick: right pre-grasp offset (descend there, then slide in); e.g. 0.07 0 0 = palm-first from the robot's right")
parser.add_argument("--r-rise", type=float, default=0.0, help="stick: first raise the right wrist this far at home orientation, then turn to the grasp orientation during reach_above (avoids sweeping the rotated hand through the stick)")
parser.add_argument("--n-close-hold", type=int, default=0, help="hold still after a hand closes before the next motion (fingers lag the command ~15 steps)")
parser.add_argument("--n-open-wait", type=int, default=25, help="stick: hold still after a hand opens, before it moves away")
parser.add_argument("--r-out-vec", type=float, nargs=3, default=(0.0, -0.12, -0.03), help="stick: right hand exit after opening (+x = away from its own fingers, mirror of the left's side entry)")
parser.add_argument("--r-park", type=float, nargs=3, default=(0.10, -0.08, 0.0), help="stick: right wrist parks at home+this after letting go (clearly released)")
parser.add_argument("--l-carry-clear", type=float, default=0.10, help="stick: carry with the lowest point this high above the table")
parser.add_argument("--l-back-vec", type=float, nargs=3, default=(0.0, -0.15, 0.0), help="stick: left backs out before returning home (fingers clear the placed stick)")
parser.add_argument("--l-out-vec", type=float, nargs=3, default=(-0.10, 0.0, 0.02), help="stick: left retreat after release (reverse of the side entry)")
parser.add_argument("--l-pre-vec", type=float, nargs=3, default=None, help="stick: left pre-grasp offset from the grasp target (e.g. -0.08 0 0 = slide in from the left)")
parser.add_argument("--l-above", type=float, default=0.12, help="stick: left pre-grasp height above its grip target")
parser.add_argument("--l-up", type=float, default=0.0, help="mirror mode: raise the left wrist straight up by this much before l_pre")
parser.add_argument("--r-yaw-deg", type=float, default=0.0, help="mirror mode: yaw the right wrist about world z after carry_mid")
parser.add_argument("--l-quat-home", type=int, default=1, help="1: left grasp uses the left home wrist quat (not the x-mirror of the right)")
parser.add_argument("--spawn-z", type=float, default=None,
                    help="override the object spawn height (m); by default it follows the measured table top")
parser.add_argument("--table-top", type=float, default=None,
                    help="override the measured packing-table top (m); default = measured")
parser.add_argument("--l-replay", action="store_true", default=False,
                    help="the LEFT hand takes the wheel with the SAME motion that works on "
                         "the right - the recorded wrist path of the right grasp replay, mirrored "
                         "back to the left arm and translated so it ends on the held wheel")
parser.add_argument("--hold-lift", type=float, default=0.10,
                    help="--l-replay: how far above its replay-expected height the right hand holds "
                         "the wheel while the left takes it (clears the table)")
parser.add_argument("--l-z-offset", type=float, default=0.04,
                    help="vertical offset of the left grasp target vs the mirrored right one (m)")
import sys  # noqa: E402
sys.path.insert(0, _HERE)
from handover_common_args import add_common_args, resolve_obj_start  # noqa: E402  (pure argparse, no isaaclab)
add_common_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
from handover_common_args import resolve_obj_mass  # noqa: E402
resolve_obj_mass(args_cli)
if args_cli.phase_scale != 1.0:                 # slower phases = smaller accelerations
    for _a in [a for a in vars(args_cli) if a.startswith("n_")]:
        setattr(args_cli, _a, max(1, int(round(getattr(args_cli, _a) * args_cli.phase_scale))))
args_cli.enable_cameras = not args_cli.no_video
if args_cli.out is None:
    args_cli.out = f"results/handover_{args_cli.robot}"
if args_cli.grasp_replay is None:      # the wheel needs the proven replayed reach; cylinder/cube use IK
    args_cli.grasp_replay = "results/dex3_batch60/ep1_actions.npy" if args_cli.object == "steering_wheel" else ""
if args_cli.stage_label is None:
    args_cli.stage_label = "A" if args_cli.robot == "dex3" else "B"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_mul  # noqa: E402

import handover_common as hc  # noqa: E402


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    scene_cfg = hc.make_scene_cfg(args_cli)
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args_cli.device))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    hc.set_stick_com(scene, args_cli)
    dt = sim.get_physics_dt()
    dev = sim.device

    robot, obj = scene["robot"], scene[f"obj_{args_cli.object}"]
    names = list(robot.data.joint_names)
    replay_order = hc.load_dex3_hand_order(args_cli.replay_actions) if args_cli.replay_actions else None
    hand = hc.HandModel(robot, args_cli.robot, args_cli.grip, dex3_hand_order=replay_order)
    ik_R = hc.ArmIK(robot, "right", dev, args_cli.ik_lambda)
    ik_L = hc.ArmIK(robot, "left", dev, args_cli.ik_lambda)
    print(f"HANDOVER_ROBOT {args_cli.robot} state_dim={hand.n_state}", flush=True)
    print(f"HANDOVER_JOINTS {len(names)} {names}", flush=True)
    print(f"HANDOVER_ARM_IDX {hand.arm_idx}", flush=True)
    print(f"HANDOVER_ARM_R {ik_R.joint_ids} ARM_L {ik_L.joint_ids}", flush=True)
    print(f"HANDOVER_HAND_R {hand.side_idx['right']} {[names[i] for i in hand.side_idx['right']]}", flush=True)
    print(f"HANDOVER_HAND_L {hand.side_idx['left']} {[names[i] for i in hand.side_idx['left']]}", flush=True)
    print(f"HANDOVER_TIPS {hand.tip_row_names()}", flush=True)
    print(f"HANDOVER_WRIST_BODIES R={ik_R.body_id} L={ik_L.body_id} "
          f"jac_shape={tuple(robot.root_physx_view.get_jacobians().shape)} "
          f"fixed_base={robot.is_fixed_base}", flush=True)
    if args_cli.robot == "brainco":
        print(f"HANDOVER_MIMIC_PAIRS {len(hand.mimic_pairs)} DEX3_ORDER {hand.dex3_hand_order}", flush=True)
        if not hc.mimic_ok(hand):
            raise SystemExit(f"expected 10 mimic pairs, got {len(hand.mimic_pairs)}")
    for i in hand.hand_idx:
        print(f"HANDOVER_LIMIT {names[i]:32s} [{hand.lim[i,0]:+.3f},{hand.lim[i,1]:+.3f}]", flush=True)
    closed = hand.hand_block_from_grip(1.0, 1.0)
    print(f"HANDOVER_CLOSED {dict(zip(hand.hand_names, closed.round(3).tolist()))}", flush=True)

    default = robot.data.default_joint_pos.clone()
    default_vel = torch.zeros_like(robot.data.default_joint_vel)
    root_z_init = float(robot.data.root_pos_w[0, 2])
    q_root = torch.tensor([hc.ROBOT_ROT], device=dev, dtype=torch.float32)
    q_fixed = quat_mul(q_root, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev))

    def settle(n):
        for _ in range(n):
            robot.set_joint_position_target(default)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

    hf = tuple(args_cli.hand_friction)
    n_mat = (hc.bind_hand_material(robot, None, None, None) if hf[0] < 0
             else hc.bind_hand_material(robot, *hf))
    if args_cli.table_top is not None:
        hc.TABLE_TOP = float(args_cli.table_top)
        print(f"HANDOVER_TABLE_TOP {hc.TABLE_TOP:.4f} (forced by --table-top)", flush=True)
    else:
        hc.measure_table_top(scene, settle, dev)
    table_top = hc.table_top_z()
    rest_z = hc.rest_z_for(args_cli.object)
    ox, oy = resolve_obj_start(args_cli)
    sz = hc.spawn_z(args_cli.object) if args_cli.spawn_z is None else float(args_cli.spawn_z)

    if args_cli.dry_run:
        hc.place_objects(scene, args_cli.object, (ox, oy), sz, dev)
        settle(5)
        print(f"HANDOVER_HOME_R {ik_R.wrist_pose_w()[0].cpu().numpy().round(4).tolist()}", flush=True)
        print(f"HANDOVER_HOME_L {ik_L.wrist_pose_w()[0].cpu().numpy().round(4).tolist()}", flush=True)
        print(f"HANDOVER_OBJ {obj.data.root_pos_w[0].cpu().numpy().round(4).tolist()}", flush=True)
        print(f"HANDOVER_TIPS_NOW {hc.tips_now(robot, hand).round(3).tolist()}", flush=True)
        hc.print_world_bboxes(["/World/envs/env_0/PackingTable", "/World/envs/env_0/ObjSteeringWheel",
                               "/World/envs/env_0/Robot/right_wrist_yaw_link"])
        print("HANDOVER_DRY_RUN_OK", flush=True)
        return

    def calibrate_pocket(n=None):
        """Close both hands at the HOME pose and measure where a grasped object sits.

        Returns (pocket_R, pocket_L): the world offset from each wrist_yaw_link to the
        centroid of that hand's three closed fingertips, at the home wrist orientation.
        Rotating the wrist by R in the world rotates this offset by R, so the IK wrist
        target for any grasp point is  grasp_point - R @ pocket.  Nothing here is
        hand-authored: it is the hand's own closed geometry.
        """
        n = int(args_cli.n_cal if n is None else n)
        t = default.clone()
        for k in range(n):
            a = (k + 1) / n
            hand.write_hand(t, hand.hand_block_from_grip(a, a))
            robot.set_joint_position_target(t)
            for _ in range(n_ctrl_phys):
                scene.write_data_to_sim(); sim.step(); scene.update(dt)
        tp = hc.tips_now(robot, hand)
        q = robot.data.joint_pos[0].cpu().numpy()
        wL, wR = tp[0].astype(np.float64), tp[1].astype(np.float64)
        nL = len(hand.tip_ids["left"]); nR = len(hand.tip_ids["right"])   # 3 for Dex3, 5 for BrainCo
        sl_L, sl_R = slice(2, 2 + nL), slice(2 + nL, 2 + nL + nR)
        pL = tp[sl_L].astype(np.float64).mean(axis=0) - wL
        pR = tp[sl_R].astype(np.float64).mean(axis=0) - wR
        print(f"HANDOVER_CAL closed tips={tp.round(3).tolist()}", flush=True)
        print(f"HANDOVER_CAL pocket_R={pR.round(4).tolist()} pocket_L={pL.round(4).tolist()} "
              f"tipsR_rel={(tp[sl_R] - wR).round(4).tolist()}", flush=True)
        print("HANDOVER_CAL closed_q " + " ".join(
            f"{names[i].replace('_joint','')}={q[i]:+.3f}" for i in hand.side_idx["right"]),
            flush=True)
        robot.write_joint_state_to_sim(default.clone(), default_vel.clone())
        robot.reset()
        settle(15)
        tp0 = hc.tips_now(robot, hand)
        print(f"HANDOVER_CAL open tipsR_rel="
              f"{(tp0[sl_R].astype(np.float64) - tp0[1].astype(np.float64)).round(4).tolist()}",
              flush=True)
        return pR, pL

    rng = np.random.default_rng(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    g_off = np.asarray(args_cli.grasp_offset, np.float64)
    g_off_L = g_off * np.array([-1.0, 1.0, 1.0])
    place = np.array([args_cli.place_xy[0], args_cli.place_xy[1], 0.0])
    mid = np.array([args_cli.mid_xy[0], args_cli.mid_xy[1], 0.0])
    n_ctrl_phys = args_cli.decimation
    tip_ref = hc.TipReference(args_cli.ref_tips_dir) if args_cli.ref_tips_dir else None

    pocket_R = pocket_L = None
    if args_cli.object == "bar" or args_cli.stick:   # replay too: the stick phases need the pocket geometry
        pocket_R, pocket_L = calibrate_pocket()
        if args_cli.pocket is not None:
            pocket_R = np.asarray(args_cli.pocket, np.float64)
            pocket_L = pocket_R * np.array([-1.0, 1.0, 1.0])
            print(f"HANDOVER_CAL pocket OVERRIDDEN by --pocket {pocket_R.round(4).tolist()}",
                  flush=True)

    replay_files = None
    if args_cli.replay_actions:
        replay_files = sorted(glob.glob(os.path.join(args_cli.replay_actions, "ep*_actions.npy")),
                              key=lambda p: int(os.path.basename(p)[2:].split("_")[0]))
        if not replay_files:
            raise SystemExit(f"no ep*_actions.npy in {args_cli.replay_actions}")
        replay_files = replay_files[:args_cli.episodes]
        print(f"HANDOVER_REPLAY {len(replay_files)} files from {args_cli.replay_actions}", flush=True)

    grasp_src = None
    if args_cli.grasp_replay and not replay_files:
        g = np.load(args_cli.grasp_replay).astype(np.float64)       # (T, 28) robot-order arms
        left7 = g[:, 0:14][:, 0::2]                                  # L cols in ARM_JOINTS order
        grasp_src = left7 * hc.ARM_MIRROR                            # -> right arm
        print(f"HANDOVER_GRASP_REPLAY {args_cli.grasp_replay} T={len(g)} mirrored", flush=True)

    n_eps = len(replay_files) if replay_files else args_cli.episodes
    episodes, n_ok = [], 0
    stage = args_cli.stage_label

    for ep in range(n_eps):
        jx, jy = rng.uniform(-args_cli.pose_noise, args_cli.pose_noise, 2)
        robot.write_joint_state_to_sim(default.clone(), default_vel.clone())
        robot.reset()
        robot.set_joint_position_target(default.clone())   # clear last episode's targets before the object is placed
        if hc.BC_OPPOSE is not None:          # right thumb swings across the palm only during reach_above
            hc.BC_SWING["R"] = hc.BC_SWING["L"] = 0.0
        robot.write_data_to_sim()
        for _ in range(10):
            sim.step(render=False); scene.update(sim.get_physics_dt())
        hc.place_objects(scene, args_cli.object, (ox + jx, oy + jy), sz, dev)
        ik_R.ctrl.reset(); ik_L.ctrl.reset()
        settle(60)
        if args_cli.obj_pose_exact is not None:
            hc.place_objects(scene, args_cli.object, args_cli.obj_pose_exact[:2],
                             args_cli.obj_pose_exact[2], dev)
            settle(3)
            print(f"ep{ep} HANDOVER_OBJ_EXACT cmd={list(args_cli.obj_pose_exact)} "
                  f"now={obj.data.root_pos_w[0].cpu().numpy().round(4).tolist()}", flush=True)

        if not args_cli.no_video:             # Isaac Sim 5.1 here sometimes starts with dead tiled cameras
            def _black():
                return [kk for kk, _ in hc.CAM_KEYS
                        if float(scene[kk].data.output["rgb"][0, ..., :3].float().mean()) < 1.0]
            _bl = _black()
            for _ in range(30):
                if not _bl:
                    break
                sim.render(); scene.update(dt); _bl = _black()
            if _bl:
                print(f"HANDOVER_BLACK_CAMERA {_bl} ep={ep}: exiting (code 3) so the caller relaunches", flush=True)
                os._exit(3)
        obj_p0 = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
        z0 = float(obj_p0[2])
        home_R = ik_R.wrist_pose_w()[0].cpu().numpy().astype(np.float64)
        home_L = ik_L.wrist_pose_w()[0].cpu().numpy().astype(np.float64)
        if args_cli.wrist_quat == "hold":
            quat_R = torch.tensor([home_R[3:7]], device=dev, dtype=torch.float32)
            quat_L = torch.tensor([home_L[3:7]], device=dev, dtype=torch.float32)
        else:
            quat_R = quat_L = q_fixed

        bar = None
        if (args_cli.object == "bar" or args_cli.stick) and pocket_R is not None:
            th = math.radians(args_cli.bar_roll_deg)
            ps = math.radians(args_cli.bar_yaw_deg)
            ct, st_ = math.cos(th), math.sin(th)
            cz_, sz_ = math.cos(ps), math.sin(ps)   # NOT `sz`: that name is the object spawn height
            RyR = np.array([[ct, 0.0, st_], [0.0, 1.0, 0.0], [-st_, 0.0, ct]])
            RzR = np.array([[cz_, -sz_, 0.0], [sz_, cz_, 0.0], [0.0, 0.0, 1.0]])
            ph = math.radians(args_cli.bar_pitch_deg)       # about world x: mirror-symmetric, same for both hands
            RxR = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(ph), -math.sin(ph)], [0.0, math.sin(ph), math.cos(ph)]])
            RR = RzR @ RyR @ RxR
            RL = RzR.T @ RyR.T @ RxR                  # mirrored: Rz(-yaw) Ry(-roll) Rx(pitch)
            qyR = torch.tensor([[math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]],
                               device=dev, dtype=torch.float32)
            qyL = torch.tensor([[math.cos(th / 2), 0.0, -math.sin(th / 2), 0.0]],
                               device=dev, dtype=torch.float32)
            qzR = torch.tensor([[math.cos(ps / 2), 0.0, 0.0, math.sin(ps / 2)]],
                               device=dev, dtype=torch.float32)
            qzL = torch.tensor([[math.cos(ps / 2), 0.0, 0.0, -math.sin(ps / 2)]],
                               device=dev, dtype=torch.float32)
            qx_ = torch.tensor([[math.cos(ph / 2), math.sin(ph / 2), 0.0, 0.0]], device=dev, dtype=torch.float32)
            quat_R = quat_mul(quat_mul(quat_mul(qzR, qyR), qx_),
                              torch.tensor([home_R[3:7]], device=dev, dtype=torch.float32))
            quat_L = quat_mul(quat_mul(quat_mul(qzL, qyL), qx_),
                              torch.tensor([home_L[3:7]], device=dev, dtype=torch.float32))
            shift = np.asarray(args_cli.wrist_shift, np.float64)
            offR, offL = RR @ pocket_R, RL @ pocket_L
            if args_cli.stick:                     # upright stick: right takes the lower part, left the upper part
                gpR = obj_p0 + np.array([0.0, 0.0, -args_cli.stick_lo])
                gpL = obj_p0 + np.array([0.0, 0.0, +args_cli.stick_hi])
            elif args_cli.bar_axis == "y":           # bar along y: right takes the near part, left the far part
                gpR = obj_p0 + np.array([0.0, -args_cli.bar_grip_x, 0.0])
                gpL = obj_p0 + np.array([0.0, +args_cli.bar_grip_x, 0.0])
            else:
                gpR = obj_p0 + np.array([+args_cli.bar_grip_x, 0.0, 0.0])
                gpL = obj_p0 + np.array([-args_cli.bar_grip_x, 0.0, 0.0])
            bar = {"gpR": gpR, "gpL": gpL, "offR": offR, "offL": offL,
                   "WR": gpR - offR + shift, "WL": gpL - offL + shift * [-1, 1, 1]}
            print(f"ep{ep} HANDOVER_BARGEO obj={obj_p0.round(4).tolist()} "
                  f"gpR={gpR.round(4).tolist()} pocketR={pocket_R.round(4).tolist()} "
                  f"offR={offR.round(4).tolist()} W_grasp_R={bar['WR'].round(4).tolist()} "
                  f"W_grasp_L={bar['WL'].round(4).tolist()} "
                  f"home_R={home_R[:3].round(4).tolist()} roll={args_cli.bar_roll_deg} "
                  f"yaw={args_cli.bar_yaw_deg} shift={shift.round(4).tolist()}",
                  flush=True)

        grasp_R = obj_p0 + g_off
        mid_obj = mid.copy(); mid_obj[2] = z0 + args_cli.mid_height
        mid_R, mid_L = mid_obj + g_off, mid_obj + g_off_L
        place_obj = place.copy(); place_obj[2] = rest_z
        place_L_high = place_obj + g_off_L + np.array([0, 0, args_cli.mid_height])
        place_L = place_obj + g_off_L
        # Geometry after the grasp (wrist<->object offset, grasp orientation) is only known
        # once the right hand holds the wheel, so the later targets are callables evaluated
        # at phase start from `G` (filled in by capture_grasp()).
        G = {"off_R": g_off.copy(), "off_L": g_off_L.copy()}

        def capture_grasp():
            wr = ik_R.wrist_pose_w()[0].cpu().numpy().astype(np.float64)
            op_ = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            G["off_R"] = wr[:3] - op_
            G["off_L"] = G["off_R"] * np.array([-1.0, 1.0, 1.0]) + np.array(
                [-args_cli.l_side_offset, args_cli.l_y_offset, args_cli.l_z_offset])
            G["quat_R"] = wr[3:7]
            G["quat_L"] = np.asarray(home_L[3:7], np.float64) if args_cli.l_quat_home else hc.mirror_quat_x(wr[3:7])
            if args_cli.l_yaw_deg:
                _th = math.radians(args_cli.l_yaw_deg); _qz = np.array([math.cos(_th / 2), 0.0, 0.0, math.sin(_th / 2)])
                _b = G["quat_L"]; G["quat_L_pre"] = np.array([
                    _qz[0]*_b[0] - _qz[3]*_b[3], _qz[0]*_b[1] - _qz[3]*_b[2],
                    _qz[0]*_b[2] + _qz[3]*_b[1], _qz[0]*_b[3] + _qz[3]*_b[0]])
            print(f"ep{ep} HANDOVER_GRASP_GEOM wrist={wr[:3].round(3).tolist()} obj={op_.round(3).tolist()} "
                  f"off_R={G['off_R'].round(3).tolist()} quat_R={wr[3:7].round(3).tolist()} "
                  f"tips={hc.tips_now(robot, hand)[5:].round(3).tolist()}", flush=True)

        f0, f1, f2 = args_cli.grasp_frames
        st = args_cli.grasp_stride
        if grasp_src is not None:
            def seg(a, b):
                fr = np.arange(a, b, st)
                return grasp_src[np.clip(fr, 0, len(grasp_src) - 1)]
            reach_tr, close_tr, lift_tr = seg(f0, f1), seg(f1, f2), seg(f2, args_cli.lift_frame)
            grasp_phases = [
                ("reach_rp", len(reach_tr), None, None, 0.0, 0.0, reach_tr),
                ("close_rp", len(close_tr), None, None, 1.0, 0.0, close_tr),
                ("lift_rp",  len(lift_tr),  None, None, 1.0, 0.0, lift_tr),
            ]
        else:
            pre = np.array([args_cli.pre_offset, 0.0, 0.0])
            grasp_phases = [
                ("reach_above", args_cli.n_reach,    grasp_R + pre + [0, 0, args_cli.reach_height], None, 0.0, 0.0, None),
                ("descend",     args_cli.n_descend,  grasp_R + pre, None, 0.0, 0.0, None),
                ("slide_in",    args_cli.n_slide,    grasp_R, None, 0.0, 0.0, None),
                ("r_close",     args_cli.n_rclose,   None, None, 1.0, 0.0, None),
                ("lift",        args_cli.n_lift,     grasp_R + [0, 0, args_cli.lift_height], None, 1.0, 0.0, None),
            ]
        if bar is not None:
            pre = np.array([0.0, -args_cli.bar_pre, 0.0])
            if args_cli.r_pre_vec is not None:     # side entry (BrainCo power grasp: palm leads onto the stick)
                pre = np.asarray(args_cli.r_pre_vec, np.float64)
            up = np.array([0.0, 0.0, args_cli.bar_above])
            grasp_phases = [
                ("reach_above", args_cli.n_reach,   bar["WR"] + pre + up, None, 0.0, 0.0, None),
                ("descend",     args_cli.n_descend, bar["WR"],            None, 0.0, 0.0, None),
                ("r_close",     args_cli.n_rclose,  bar["WR"] + np.asarray(args_cli.r_close_advance), None, 1.0, 0.0, None),
            ] + ([("r_hold", args_cli.n_close_hold, None, None, 1.0, 0.0, None)] if args_cli.n_close_hold else []) + [
                ("lift",        args_cli.n_lift,    bar["WR"] + np.asarray(args_cli.r_close_advance) + [0, 0, args_cli.bar_lift],
                 None, 1.0, 0.0, None),
            ]
            if args_cli.n_slide > 0 and (args_cli.bar_pre > 0 or args_cli.r_pre_vec is not None):   # optional horizontal entry
                grasp_phases.insert(2, ("slide_in", args_cli.n_slide, bar["WR"],
                                        None, 0.0, 0.0, None))
                grasp_phases[1] = ("descend", args_cli.n_descend, bar["WR"] + pre,
                                   None, 0.0, 0.0, None)
            if args_cli.r_rise > 0:                # rise at home orientation, then rotate while moving over
                grasp_phases[0] = tuple(grasp_phases[0][:7]) + (("R", args_cli.bar_roll_deg),)
                grasp_phases.insert(0, ("r_rise", 30, home_R[:3] + np.array([0.0, 0.0, args_cli.r_rise]),
                                        None, 0.0, 0.0, None))

        preL = np.array(args_cli.pre_vec, np.float64) if args_cli.pre_vec else np.array([-args_cli.pre_offset, 0.0, 0.0])
        preR = np.array([args_cli.pre_offset, 0.0, 0.0])
        def wR_now():
            return ik_R.wrist_pose_w()[0, :3].cpu().numpy().astype(np.float64)

        def obj_now():
            return obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)

        def wL_now():
            return ik_L.wrist_pose_w()[0, :3].cpu().numpy().astype(np.float64)

        def left_grasp_target():
            # wheel hangs from the right hand after the roll: grab the rim point opposite the
            # right hand (the lowest point), pinching the tube along x
            op_ = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            d = op_ - wR_now(); d /= max(np.linalg.norm(d), 1e-6)
            rim = op_ + d * hc.WHEEL_RADIUS
            G["L_grasp"] = rim + np.asarray(args_cli.l_grasp_offset)
            print(f"ep{ep} HANDOVER_HANG obj={op_.round(3).tolist()} wR={wR_now().round(3).tolist()} "
                  f"rim_opp={rim.round(3).tolist()} L_grasp={G['L_grasp'].round(3).tolist()}", flush=True)
            return G["L_grasp"]

        rp_wrist, rp_obj = [], []          # right wrist pose / object pos during the replay grasp
        n_reach_rp = len(reach_tr) if grasp_src is not None else 0
        n_close_rp = len(close_tr) if grasp_src is not None else 0

        def build_left_path():
            """Cache the mirrored+translated left path once, at l_pre, from the live wheel."""
            reach = mirrored_left_path(0, n_reach_rp)
            close = mirrored_left_path(n_reach_rp, n_reach_rp + n_close_rp)
            G["l_path_reach"], G["l_path_close"] = reach, close
            return reach

        def mirrored_left_path(i0, i1):
            """The recorded RIGHT replay wrist path [i0:i1], mirrored to the left arm and
            translated so the (mirrored) wheel lands on the wheel the right hand is holding
            right now. Same motion, same sweep, same closure - just moved to the wheel."""
            path = np.asarray(rp_wrist[i0:i1], np.float64).copy()
            hub_ref = np.asarray(rp_obj[min(n_reach_rp - 1, len(rp_obj) - 1)], np.float64)
            mirror = np.array([-1.0, 1.0, 1.0])
            delta = obj_now() - hub_ref * mirror
            out = np.zeros((len(path), 7), np.float64)
            out[:, :3] = path[:, :3] * mirror + delta
            for i, q in enumerate(path[:, 3:7]):
                out[i, 3:7] = hc.mirror_quat_x(q)
            G["l_delta"] = delta
            print(f"ep{ep} HANDOVER_LSWEEP i=[{i0},{i1}) hub_ref={hub_ref.round(3).tolist()} "
                  f"obj_now={obj_now().round(3).tolist()} delta={delta.round(3).tolist()} "
                  f"start={out[0, :3].round(3).tolist()} end={out[-1, :3].round(3).tolist()}",
                  flush=True)
            return out

        def left_carry_offset():
            op_ = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            G["off_L2"] = wL_now() - op_
            print(f"ep{ep} HANDOVER_LEFT_HOLD off_L2={G['off_L2'].round(3).tolist()}", flush=True)
            return G["off_L2"]

        # (name, n, end_R, end_L, grip_R_end, grip_L_end, arm_traj, rot)   rot = (side, deg about world y)
        if args_cli.phases == "grasp_only":
            phases = grasp_phases + [
                ("hold_still", args_cli.n_hold_still, lambda: wR_now(), None, 1.0, 0.0, None),
            ]
        elif args_cli.l_replay:
            phases = grasp_phases + [
                ("carry_mid",   args_cli.n_carry,    lambda: mid_obj + G["off_R"], None, 1.0, 0.0, None),
                ("hold",        args_cli.n_hold,     lambda: wR_now(), None, 1.0, 0.0, None),
                ("l_pre",       args_cli.n_lpre,     None, lambda: build_left_path()[0][:3], 1.0, 0.0, None),
                ("l_sweep",     n_reach_rp,          None, None, 1.0, 0.0, None, None,
                 lambda: G["l_path_reach"]),
                ("l_close",     n_close_rp,          None, None, 1.0, 1.0, None, None,
                 lambda: G["l_path_close"]),
                ("r_open",      args_cli.n_ropen,    None, None, 0.0, 1.0, None),
                ("r_out",       args_cli.n_out,      lambda: wR_now() + np.asarray(args_cli.r_lift_away), lambda: wL_now(), 0.0, 1.0, None),
                ("r_retract",   args_cli.n_rretract, home_R[:3], lambda: wL_now(), 0.0, 1.0, None),
                ("l_carry",     args_cli.n_lcarry,   None, lambda: place_obj + left_carry_offset() + [0, 0, args_cli.mid_height], 0.0, 1.0, None),
                ("lower",       args_cli.n_lower,    None, lambda: place_obj + G["off_L2"] + [0, 0, 0.005], 0.0, 1.0, None),
                ("l_open",      args_cli.n_lopen,    None, None, 0.0, 0.0, None),
                ("l_out",       args_cli.n_out,      None, lambda: wL_now() + [-0.06, -0.10, 0.05], 0.0, 0.0, None),
                ("l_retract",   args_cli.n_lretract, None, home_L[:3], 0.0, 0.0, None),
            ]
        elif args_cli.handover == "mirror":
            recv = np.array(args_cli.l_receive, np.float64) if args_cli.l_receive else None
            pre_phases = ([("l_wait", args_cli.n_lpre, None, recv, 0.0, 0.0, None)] if recv is not None else [])
            phases = (grasp_phases + pre_phases if args_cli.l_wait_after_lift else pre_phases + grasp_phases) + [
                ("carry_mid",   args_cli.n_carry,    (lambda: recv - G["off_L"] + G["off_R"] - np.array([0.0, args_cli.recv_approach, 0.0])) if recv is not None else (lambda: mid_obj + G["off_R"]), None, 1.0, 0.0, None),
            ] + ([("r_yaw", args_cli.n_roll, lambda: wR_now(), None, 1.0, 0.0, None, ("R", args_cli.r_yaw_deg, "z"))]
                 if args_cli.r_yaw_deg else []) + ([("l_up", args_cli.n_lpre, None, lambda: wL_now() + [0.0, 0.0, args_cli.l_up], 1.0, 0.0, None)]
                 if args_cli.l_up else []) + ([("r_in", 40, lambda: wR_now() + [0.0, args_cli.recv_approach, 0.0], None, 1.0, 0.0, None)]
                 if (recv is not None and args_cli.recv_approach) else []) + [
                ("l_pre",       args_cli.n_lpre,     None, lambda: obj_now() + G["off_L"] + preL, 1.0, 0.0, None),
                ("l_reach",     args_cli.n_lreach,   None, lambda: obj_now() + G["off_L"], 1.0, 0.0, None),
                ("l_close",     args_cli.n_lclose,   None, None, 1.0, 1.0, None),
                ("r_open",      args_cli.n_ropen,    None, None, 0.0, 1.0, None),
                ("r_out",       args_cli.n_out,      lambda: wR_now() + [0.03, -0.15, -0.05], lambda: wL_now(), 0.0, 1.0, None),
                ("r_retract",   args_cli.n_rretract, home_R[:3], lambda: wL_now(), 0.0, 1.0, None),
                ("l_carry",     args_cli.n_lcarry,   None, lambda: place_obj + G["off_L"] + [0, 0, args_cli.mid_height], 0.0, 1.0, None),
                ("lower",       args_cli.n_lower,    None, lambda: place_obj + G["off_L"] + [0, 0, 0.005], 0.0, 1.0, None),
                ("l_open",      args_cli.n_lopen,    None, None, 0.0, 0.0, None),
                ("l_out",       args_cli.n_out,      None, lambda: wL_now() + [-0.06, -0.10, 0.05], 0.0, 0.0, None),
                ("l_retract",   args_cli.n_lretract, None, home_L[:3], 0.0, 0.0, None),
            ]
        elif args_cli.handover == "stick":
            # Upright stick. Both hands use the same neutral-wrist straddle grasp from ABOVE: the stick
            # passes between the thumb (behind) and the two fingers (outer side), then the hand closes.
            # Right holds the lower part, the left comes down onto the upper part, the right opens and
            # backs away (-y, never sideways into the thumb), the left sets the stick upright on the target.
            def st_left(c):
                return c + np.array([0.0, 0.0, +args_cli.stick_hi]) - bar["offL"]
            def stick_R():
                q = obj.data.root_quat_w[0].cpu().numpy().astype(np.float64)   # w, x, y, z
                w, x, y, z = q
                return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                                 [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                                 [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
            def st_left_live(above=0.0):
                # upper grip point from the stick's ACTUAL pose (it tilts in the right hand), approached
                # along the stick axis so the stick stays in the thumb-finger gap all the way down
                Rm = stick_R(); ax = Rm @ np.array([0.0, 0.0, 1.0])
                if ax[2] < 0: ax = -ax
                gp = obj_now() + Rm @ np.array([0.0, 0.0, +args_cli.stick_hi])
                if above == 0.0:
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, ax[2]))))
                    print(f"ep{ep} HANDOVER_TILT tilt_deg={tilt:.1f} axis={ax.round(3).tolist()} gpL={gp.round(3).tolist()}", flush=True)
                if above != 0.0 and args_cli.l_pre_vec is not None:     # side entry instead of from above
                    return gp - bar["offL"] + np.asarray(args_cli.l_shift) + np.asarray(args_cli.l_pre_vec)
                return gp - bar["offL"] + np.asarray(args_cli.l_shift) + ax * above
            def st_place(dz):
                if not stick_held_by_left():
                    return wL_now()
                # left wrist target that puts the held stick's LOWEST point dz above the table at the target
                Rm = stick_R(); ax = Rm @ np.array([0.0, 0.0, 1.0])
                if ax[2] < 0: ax = -ax
                c = obj_now()
                low = c[2] - (0.5 * args_cli.stick_h * ax[2] + 0.5 * args_cli.stick_w * math.sqrt(max(0.0, 1 - ax[2] ** 2)))
                tgt_c = np.array([place_obj[0], place_obj[1], c[2] - low + hc.table_top_z() + dz])
                return wL_now() + (tgt_c - c)
            def stick_held_by_left():
                ax = stick_R() @ np.array([0.0, 0.0, 1.0])
                return abs(ax[2]) > 0.7 and np.linalg.norm(obj_now() - wL_now()) < 0.25
            def st_straight():
                # base resting on the table: move the left grip point straight above the base so the stick
                # stands vertical (it pivots on its base) before release; a 14 deg lean topples it otherwise
                Rm = stick_R(); ax = Rm @ np.array([0.0, 0.0, 1.0])
                if ax[2] < 0: ax = -ax
                c = obj_now(); base = c - 0.5 * args_cli.stick_h * ax
                if not stick_held_by_left():            # dropped: do not chase a lying stick
                    return wL_now()
                # pivot about the base using where the hand ACTUALLY grips (the stick can slide/turn in the hand)
                pocket_w = wL_now() + bar["offL"]
                d = float((pocket_w - base) @ ax)           # grip height along the stick
                # rotate the stick about its base to vertical: the axis point at the grip moves by d*(e_z - ax).
                # Only that displacement; the hand's offset from the axis must stay as it is (earlier version
                # also pulled the nominal grasp point onto the axis and dragged the stick sideways).
                tgt = wL_now() + d * (np.array([0.0, 0.0, 1.0]) - ax)
                lowest = c[2] - (0.5 * args_cli.stick_h * ax[2] + 0.5 * args_cli.stick_w * math.sqrt(max(0.0, 1 - ax[2] ** 2)))
                if lowest > hc.table_top_z() + 0.012:      # base not on the table: pivoting would just swing it
                    print(f"ep{ep} HANDOVER_STRAIGHTEN skipped: lowest point {lowest - hc.table_top_z():.3f} above table", flush=True)
                    return wL_now()
                mv = tgt - wL_now(); n = float(np.linalg.norm(mv))
                if n > args_cli.straighten_max:            # small re-measured passes instead of one big swing
                    tgt = wL_now() + mv * (args_cli.straighten_max / n)
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, ax[2]))))
                print(f"ep{ep} HANDOVER_STRAIGHTEN tilt_deg={tilt:.1f} base={base.round(3).tolist()} d={d:.3f} move={(tgt - wL_now()).round(3).tolist()}", flush=True)
                return tgt
            phases = grasp_phases + [
                ("carry_mid",   args_cli.n_carry,    lambda: mid_obj + G["off_R"], None, 1.0, 0.0, None),
                ("hold_still",  20,                  None, None, 1.0, 0.0, None),
                ("l_pre",       args_cli.n_lpre,     None, lambda: st_left_live(args_cli.l_above), 1.0, 0.0, None),
                ("l_reach",     args_cli.n_lreach,   None, lambda: st_left_live(0.0), 1.0, 0.0, None),
                ("l_close",     args_cli.n_lclose,   None, lambda: wL_now() + np.asarray(args_cli.l_close_advance), 1.0, 1.0, None),
            ] + ([("l_hold", args_cli.n_close_hold, None, None, 1.0, 1.0, None)] if args_cli.n_close_hold else []) + [
                ("r_open",      args_cli.n_ropen,    None, None, 0.0, 1.0, None),
                ("r_wait",      args_cli.n_open_wait, None, None, 0.0, 1.0, None),   # fingers lag the command ~15 steps
                ("r_out",       args_cli.n_out,      lambda: wR_now() + np.asarray(args_cli.r_out_vec), None, 0.0, 1.0, None),
                ("r_retract",   args_cli.n_rretract, home_R[:3] + np.asarray(args_cli.r_park), None, 0.0, 1.0, None),
                ("l_carry",     args_cli.n_lcarry,   None, lambda: st_place(args_cli.l_carry_clear), 0.0, 1.0, None),
                ("lower",       args_cli.n_lower,    None, lambda: st_place(args_cli.lower_clear), 0.0, 1.0, None),
                ("l_straighten", 25,                 None, lambda: st_straight(), 0.0, 1.0, None),
                ("l_straighten2", 25,                None, lambda: st_straight(), 0.0, 1.0, None),   # re-measured: the base can slide
                ("l_straighten3", 25,                None, lambda: st_straight(), 0.0, 1.0, None),
                ("l_open",      args_cli.n_lopen + 10, None, None, 0.0, 0.0, None),
                ("l_wait",      args_cli.n_open_wait, None, None, 0.0, 0.0, None),
                ("l_out",       args_cli.n_out,      None, lambda: wL_now() + np.asarray(args_cli.l_out_vec), 0.0, 0.0, None),
                ("l_back",      args_cli.n_out,      None, lambda: wL_now() + np.asarray(args_cli.l_back_vec), 0.0, 0.0, None),
                ("settle",      40,                  None, None, 0.0, 0.0, None),   # 2 s: the placed stick must stay put
            ]
        elif args_cli.handover == "bar":
            # Bar along y held at its near part by the right hand.  After the carry the left takes the
            # FAR part (+y) in mid-air with the same neutral-wrist pinch as the right hand, then the
            # right opens and moves away, and the left places the bar on the left side of the table.
            def bar_left_target():
                return obj_now() + np.array([0.0, +args_cli.bar_grip_far, 0.0]) - bar["offL"]
            def bar_place_target():
                return place_obj + np.array([0.0, +args_cli.bar_grip_far, 0.0]) - bar["offL"]
            phases = grasp_phases + [
                ("carry_mid",   args_cli.n_carry,    lambda: mid_obj + G["off_R"], None, 1.0, 0.0, None),
                ("hold_still",  30,                  None, None, 1.0, 0.0, None),
                ("l_pre",       args_cli.n_lpre,     None, lambda: bar_left_target() + [0.0, 0.0, 0.10], 1.0, 0.0, None),
                ("l_reach",     args_cli.n_lreach,   None, lambda: bar_left_target(), 1.0, 0.0, None),
                ("l_close",     args_cli.n_lclose,   None, None, 1.0, 1.0, None),
                ("r_open",      args_cli.n_ropen,    None, None, 0.0, 1.0, None),
                ("r_out",       args_cli.n_out,      lambda: wR_now() + [0.0, -0.10, 0.06], None, 0.0, 1.0, None),
                ("r_retract",   args_cli.n_rretract, home_R[:3], None, 0.0, 1.0, None),
                ("l_carry",     args_cli.n_lcarry,   None, lambda: bar_place_target() + [0.0, 0.0, 0.10], 0.0, 1.0, None),
                ("lower",       args_cli.n_lower,    None, lambda: bar_place_target() + [0.0, 0.0, 0.01], 0.0, 1.0, None),
                ("l_open",      args_cli.n_lopen,    None, None, 0.0, 0.0, None),
                ("l_out",       args_cli.n_out,      None, lambda: wL_now() + [0.0, 0.0, 0.12], 0.0, 0.0, None),
                ("l_retract",   args_cli.n_lretract, None, home_L[:3], 0.0, 0.0, None),
            ]
        else:
          phases = grasp_phases + [
            ("carry_mid",   args_cli.n_carry,    lambda: mid_obj + G["off_R"], None, 1.0, 0.0, None),
            ("r_roll",      args_cli.n_roll,     lambda: mid_obj + G["off_R"], None, 1.0, 0.0, None, ("R", args_cli.roll_deg)),
            ("l_pre",       args_cli.n_lpre,     None, lambda: left_grasp_target() + np.asarray(args_cli.l_pre_offset), 1.0, 0.0, None),
            ("l_reach",     args_cli.n_lreach,   None, lambda: G["L_grasp"], 1.0, 0.0, None),
            ("l_close",     args_cli.n_lclose,   None, None, 1.0, 1.0, None),
            ("r_open",      args_cli.n_ropen,    None, None, 0.0, 1.0, None),
            ("r_out",       args_cli.n_out,      lambda: wR_now() + np.asarray(args_cli.r_lift_away), None, 0.0, 1.0, None),
            ("r_up",        args_cli.n_rretract, lambda: np.array([home_R[0], home_R[1], home_R[2] + 0.45]), None, 0.0, 1.0, None),
            ("l_roll",      args_cli.n_roll,     None, None, 0.0, 1.0, None, ("L", args_cli.roll_deg)),
            ("l_carry",     args_cli.n_lcarry,   None, lambda: place_obj + left_carry_offset() + [0, 0, 0.12], 0.0, 1.0, None),
            ("lower",       args_cli.n_lower,    None, lambda: place_obj + G["off_L2"] + [0, 0, 0.01], 0.0, 1.0, None),
            ("l_open",      args_cli.n_lopen,    None, None, 0.0, 0.0, None),
            ("l_out",       args_cli.n_out,      None, lambda: wL_now() + [0, 0, 0.12], 0.0, 0.0, None),
            ("l_retract",   args_cli.n_lretract, None, home_L[:3], 0.0, 0.0, None),
          ]
        phases = [tuple(ph) + (None,) * (9 - len(ph)) for ph in phases]
        phase_names = [p[0] for p in phases]
        lower_phase = phase_names.index("lower") if "lower" in phase_names else None

        # The replay has no phases: build one pseudo-phase per recorded chunk so the
        # per-phase wrist error report keeps the same shape (wrist targets are the
        # scripted schedule, which replay should follow if the retarget is faithful).
        replay = np.load(replay_files[ep]) if replay_files else None
        if replay is not None:
            if replay.shape[1] != 28:
                raise SystemExit(f"{replay_files[ep]}: expected (T, 28), got {replay.shape}")
            extra = len(replay) - sum(p[1] for p in phases)
            if extra > 0:                       # recorded with longer phases: extend the last one
                nm, n_, eR, eL, gR, gL, tr, ro, lt = phases[-1]
                phases[-1] = (nm, n_ + extra, eR, eL, gR, gL, tr, ro, lt)

        targets = default.clone()
        tgt_R, tgt_L = home_R[:3].copy(), home_L[:3].copy()
        grip_R, grip_L = 0.0, 0.0
        actions, states, tips, objpose = [], [], [], []
        buf = {k: [] for _, k in hc.CAM_KEYS}
        _rt = {}
        if args_cli.stick:   # a placed stick may stay upright or tip over: any pose resting ON the table counts
            _top = hc.table_top_z(); _lo = _top + args_cli.stick_w / 2; _hi = _top + args_cli.stick_h / 2
            _rt = {"rest_z": 0.5 * (_lo + _hi), "rest_tol": 0.5 * (_hi - _lo) + 0.01}
        tracker = hc.EpisodeTracker(z0, place[:2], root_z_init, args_cli.vel_spike, lower_phase,
                                    place_radius=args_cli.place_radius,
                                    phase_names=phase_names, **({"rest_z": rest_z} | _rt))
        tracker.check_spawn(obj_p0, log_prefix=f"ep{ep} ")
        # step index (1-based) each phase covers, for the offline --rescore gate check
        phase_bounds, _b = [], 0
        for _nm, _n in [(p_[0], p_[1]) for p_ in phases]:
            phase_bounds.append([_nm, _b + 1, _b + _n]); _b += _n
        track = {}
        step = 0
        t_wall0 = time.perf_counter()

        for pi, (pname, n, end_R, end_L, gR_end, gL_end, arm_tr, rot, l_traj) in enumerate(phases):
            if callable(l_traj):
                l_traj = l_traj()
            if pname == "r_rise":                    # hold the home orientation until reach_above turns the wrist
                G["quat_R_final"] = quat_R.clone()
                quat_R = torch.tensor([home_R[3:7]], device=dev, dtype=torch.float32)
            if pname == "descend" and "quat_R_final" in G:
                quat_R = G["quat_R_final"]
            if rot is not None:
                rot_base = (quat_R if rot[0] == "R" else quat_L).clone()
            if pname in ("carry_mid", "hold_still"):
                capture_grasp()                       # right hand holds the wheel: fix geometry
                if grasp_src is not None:             # joint-space grasp: keep its wrist orientation
                    quat_R = torch.tensor([G["quat_R"]], device=dev, dtype=torch.float32)
                    quat_L = torch.tensor([G["quat_L"]], device=dev, dtype=torch.float32)
            if pname == "l_pre" and args_cli.l_quat_late and "quat_L_pre" in G:
                quat_L = torch.tensor([G["quat_L_pre"]], device=dev, dtype=torch.float32)
            if callable(end_R):
                end_R = end_R()
            if callable(end_L):
                end_L = end_L()
            if arm_tr is not None:                    # joint-space phase: track the measured wrist
                tgt_R = ik_R.wrist_pose_w()[0, :3].cpu().numpy().astype(np.float64)
            start_R, start_L = tgt_R.copy(), tgt_L.copy()
            gR0, gL0 = grip_R, grip_L
            errs_R, errs_L = [], []
            for k in range(n):
                a = (k + 1) / n
                if hc.BC_OPPOSE is not None and pname == "reach_above":
                    hc.BC_SWING["R"] = a
                if hc.BC_OPPOSE is not None and pname == "l_pre":
                    hc.BC_SWING["L"] = a
                if rot is not None:                   # slerp-free: world-y rotation ramped linearly
                    th = math.radians(rot[1]) * a
                    if len(rot) > 2 and rot[2] == "z":
                        q_y = torch.tensor([[math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]], device=dev)
                    else:
                        q_y = torch.tensor([[math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]], device=dev)
                    if rot[0] == "R":
                        quat_R = quat_mul(q_y, rot_base)
                    else:
                        quat_L = quat_mul(q_y, rot_base)
                if end_R is not None:
                    tgt_R = start_R + (np.asarray(end_R) - start_R) * a
                if end_L is not None:
                    tgt_L = start_L + (np.asarray(end_L) - start_L) * a
                if l_traj is not None:                # left follows the mirrored proven path
                    row_l = l_traj[min(k, len(l_traj) - 1)]
                    tgt_L = row_l[:3].astype(np.float64)
                    quat_L = torch.tensor([row_l[3:7]], device=dev, dtype=torch.float32)
                grip_R, grip_L = hc.lerp(gR0, gR_end, a), hc.lerp(gL0, gL_end, a)

                if replay is not None:
                    if step >= len(replay):
                        break
                    row = replay[step].astype(np.float64)
                    hand.write_arms(targets, row[:14])
                    hand.write_hand(targets, hand.hand_block_from_dex3(row[14:28]))
                else:
                    if arm_tr is not None:
                        targets[0, ik_R.joint_ids] = torch.as_tensor(arm_tr[k], device=dev, dtype=torch.float32)
                        wp = ik_R.wrist_pose_w()[0].cpu().numpy().astype(np.float64)
                        tgt_R = wp[:3].copy()
                        rp_wrist.append(wp)
                        rp_obj.append(obj.data.root_pos_w[0].cpu().numpy().astype(np.float64))
                    else:
                        qR = ik_R.solve(torch.tensor([tgt_R], device=dev, dtype=torch.float32), quat_R)
                        targets[0, ik_R.joint_ids] = qR[0]
                    qL = ik_L.solve(torch.tensor([tgt_L], device=dev, dtype=torch.float32), quat_L)
                    targets[0, ik_L.joint_ids] = qL[0]
                    hand.write_hand(targets, hand.hand_block_from_grip(grip_R, grip_L))
                robot.set_joint_position_target(targets)
                for _ in range(n_ctrl_phys):
                    scene.write_data_to_sim(); sim.step(); scene.update(dt)
                step += 1

                # --- record ---
                q = robot.data.joint_pos[0].cpu().numpy()
                states.append(hand.state(q))
                actions.append(hand.action_vec(targets[0].cpu().numpy()))
                tips.append(hc.tips_now(robot, hand))
                objpose.append(np.concatenate([obj.data.root_pos_w[0].cpu().numpy(),
                                               obj.data.root_quat_w[0].cpu().numpy()]).astype(np.float32))
                if not args_cli.no_video:
                    hc.grab_frames(scene, buf)

                # --- staged metrics ---
                op = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
                ov = obj.data.root_lin_vel_w[0].cpu().numpy().astype(np.float64)
                pR = robot.data.body_pos_w[0, ik_R.body_id].cpu().numpy().astype(np.float64)
                pL = robot.data.body_pos_w[0, ik_L.body_id].cpu().numpy().astype(np.float64)
                errs_R.append(float(np.linalg.norm(tgt_R - pR)))
                errs_L.append(float(np.linalg.norm(tgt_L - pL)))
                tracker.update(op, ov, pR, pL, float(robot.data.root_pos_w[0, 2]),
                               hand.limit_violation(q), phase_index=pi, phase_name=pname,
                               log_prefix=f"ep{ep} ")
            track[pname] = {"right": float(np.mean(errs_R)) if errs_R else None,
                            "left": float(np.mean(errs_L)) if errs_L else None}
            print(f"HANDOVER_PHASE ep{ep} {pname:12s} steps={n:3d} "
                  f"errR={track[pname]['right']} errL={track[pname]['left']} "
                  f"obj_dz={float(obj.data.root_pos_w[0, 2]) - z0:+.3f} "
                  f"obj={obj.data.root_pos_w[0].cpu().numpy().round(3).tolist()} "
                  f"wR={ik_R.wrist_pose_w()[0, :3].cpu().numpy().round(3).tolist()} "
                  f"wL={ik_L.wrist_pose_w()[0, :3].cpu().numpy().round(3).tolist()}", flush=True)
            if pname in ("r_close", "close_rp", "l_close", "slide_in"):
                qn = robot.data.joint_pos[0].cpu().numpy()
                side = "left" if pname == "l_close" else "right"
                lo = hand.lim[hand.side_idx[side], 0]; hi = hand.lim[hand.side_idx[side], 1]
                jam = [names[i] for j, i in enumerate(hand.side_idx[side])
                       if min(abs(qn[i] - lo[j]), abs(qn[i] - hi[j])) < 0.02]
                print(f"HANDOVER_FINGERS ep{ep} {pname} {side} " + " ".join(
                    f"{names[i].replace(side + '_hand_', '').replace('_joint', '')}="
                    f"{qn[i]:+.3f}(cmd{float(targets[0, i]):+.3f})" for i in hand.side_idx[side])
                    + f" at_limit={jam}", flush=True)
                wr_ = ik_R.wrist_pose_w()[0].cpu().numpy()
                op_ = obj.data.root_pos_w[0].cpu().numpy()
                tp_ = hc.tips_now(robot, hand)
                print(f"HANDOVER_CONTACT ep{ep} {pname} wristR={wr_[:3].round(4).tolist()} "
                      f"quatR={wr_[3:7].round(3).tolist()} obj={op_.round(4).tolist()} "
                      f"tipsR_rel_obj={(tp_[2 + len(hand.tip_ids['left']):] - op_).round(4).tolist()}", flush=True)
        wall = time.perf_counter() - t_wall0

        op = obj.data.root_pos_w[0].cpu().numpy()
        d_place = float(np.linalg.norm(op[:2] - place[:2]))
        print(f"ep{ep} HANDOVER_FINAL obj={op.round(3).tolist()} dist_to_target_xy={d_place:.3f} "
              f"obj_dz={float(op[2]) - z0:+.3f} place_radius={args_cli.place_radius}", flush=True)
        if args_cli.phases == "grasp_only":
            wR_ = ik_R.wrist_pose_w()[0, :3].cpu().numpy().astype(np.float64)
            dz = float(op[2]) - z0
            dwr = float(np.linalg.norm(op.astype(np.float64) - wR_))
            held = dwr <= 0.10
            ok = bool(dz >= 0.08 and held)
            print(f"HANDOVER_GRASPONLY ep{ep} PASS={ok} obj_dz={dz:+.4f} (need >=0.080) "
                  f"dist_to_wristR={dwr:.4f} (need <=0.100) held={held} "
                  f"obj={op.round(4).tolist()} wristR={wR_.round(4).tolist()} z0={z0:.4f} "
                  f"t_grasp={tracker.t_grasp} dropped={tracker.dropped}", flush=True)
        tips_arr = np.array(tips, dtype=np.float32)
        rec = {"episode": ep, "stage": stage, "robot": args_cli.robot, "object": args_cli.object,
               "obj_start": obj_p0.round(4).tolist(), "obj_end": op.round(4).tolist(), "z0": z0,
               **tracker.record(),
               "failure": tracker.failure_category(),
               "phase_bounds": phase_bounds,
               "wrist_track_err": track,
               "wrist_err_mean": float(np.mean([v for d in track.values() for v in d.values()
                                                if v is not None])),
               "fingertip_err": tip_ref.error(tips_arr, hand, ep) if tip_ref else None,
               "completion_s": (tracker.t_place * args_cli.decimation / 120.0
                                if tracker.t_place is not None else None),
               "final_dist_to_target_xy": d_place, "wall_s": wall, "sim_hz_achieved": step / max(wall, 1e-6),
               "root_z": float(robot.data.root_pos_w[0, 2]),
               "replay_file": replay_files[ep] if replay_files else None}
        n_ok += int(tracker.success)
        episodes.append(rec)
        np.save(os.path.join(args_cli.out, f"ep{ep}_actions.npy"), np.array(actions))
        np.save(os.path.join(args_cli.out, f"ep{ep}_states.npy"), np.array(states))
        np.save(os.path.join(args_cli.out, f"ep{ep}_tips.npy"), tips_arr)
        np.save(os.path.join(args_cli.out, f"ep{ep}_obj.npy"), np.asarray(objpose))   # (T, 7) pos + quat wxyz, diagnostics
        with open(os.path.join(args_cli.out, f"ep{ep}_{stage}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        if not args_cli.no_video:
            hc.write_videos(buf, args_cli.out, f"demo_{ep:06d}", args_cli.video_fps)
        print(f"HANDOVER_JL ep{ep} " + str(hand.jl_report()), flush=True)
        print(f"HANDOVER_EP {ep} stage={stage} grasp={tracker.t_grasp} transfer={tracker.t_transfer} transfer_proxy={tracker.t_transfer_proxy} "
              f"place={tracker.t_place} dropped={tracker.dropped} fell={tracker.fell} "
              f"jl={tracker.jl_viol} spikes={tracker.contact_spikes} success={tracker.success} "
              f"final_near={tracker.final_near_xy} final_rest={tracker.final_resting} "
              f"steps={step}", flush=True)

    with open(os.path.join(args_cli.out, "handover_summary.json"), "w") as f:
        json.dump({"task": f"bimanual handover right->left, {args_cli.robot} G1 (Stage {stage})",
                   "stage": stage, "robot": args_cli.robot, "object": args_cli.object,
                   "mode": "replay" if replay_files else args_cli.mode,
                   "replay_actions": args_cli.replay_actions,
                   "episodes": len(episodes), "successes": n_ok,
                   "success_rate": n_ok / max(1, len(episodes)),
                   "control_hz": 120.0 / args_cli.decimation, "decimation": args_cli.decimation,
                   "phase_steps": {p: getattr(args_cli, f"n_{p}") for p in
                                   ("reach", "descend", "rclose", "lift", "carry", "lreach",
                                    "lclose", "ropen", "rretract", "lcarry", "lower",
                                    "lopen", "lretract")},
                   "obj_start_xy": list(args_cli.obj_start), "place_xy": list(args_cli.place_xy),
                   "place_radius": args_cli.place_radius, "handover": args_cli.handover,
                   "mid_height": args_cli.mid_height, "n_roll": args_cli.n_roll,
                   "grasp_replay": args_cli.grasp_replay,
                   "obj_mass": args_cli.obj_mass, "obj_friction": list(args_cli.obj_friction),
                   "table_top_measured": table_top, "rest_z": rest_z,
                   "hand_material_prims": n_mat, "spawn_z": sz,
                   "hand_friction": (None if hf[0] < 0 else list(hf)),
                   "obj_pose_exact": (list(args_cli.obj_pose_exact)
                                      if args_cli.obj_pose_exact is not None else None),
                   "l_replay": args_cli.l_replay, "hold_lift": args_cli.hold_lift,
                   "phases_mode": args_cli.phases, "bar_roll_deg": args_cli.bar_roll_deg,
                   "bar_yaw_deg": args_cli.bar_yaw_deg,
                   "bar_grip_x": args_cli.bar_grip_x, "bar_pre": args_cli.bar_pre,
                   "bar_lift": args_cli.bar_lift, "wrist_shift": list(args_cli.wrist_shift),
                   "pocket_R": (None if pocket_R is None else np.asarray(pocket_R).round(4).tolist()),
                   "phase_scale": args_cli.phase_scale,
                   "grip": args_cli.grip, "pose_noise": args_cli.pose_noise, "seed": args_cli.seed,
                   "wrist_quat": args_cli.wrist_quat, "grasp_offset": list(args_cli.grasp_offset),
                   "action_dim": hand.n_state, "arm_joint_ids": hand.arm_idx,
                   "hand_joint_ids": hand.hand_idx, "hand_joint_names": hand.hand_names,
                   "hand_idx_L": hand.side_idx["left"], "hand_idx_R": hand.side_idx["right"],
                   "dex3_hand_order": hand.dex3_hand_order,
                   "tip_rows": hand.tip_row_names(), "joint_names": names,
                   "per_episode": episodes}, f, indent=2)
    print(f"HANDOVER_SUCCESS_RATE {n_ok}/{len(episodes)}", flush=True)
    print("HANDOVER_OK", flush=True)


if __name__ == "__main__":
    main()
    # Isaac Sim (Kit) can hang in close(); exit directly once all results are written.
    # Results are already written and flushed, so exit hard.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
