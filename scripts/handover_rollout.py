r"""Unified policy client for the bimanual handover task (Stages A, B-with-policy, C).

Replaces dex3_slot_rollout.py for the handover task. Talks to policy_bridge_server.py
(pickle-over-TCP: {"op": "act", "state": {"arms": (1,14), "hands": (1,H)}, "video":
{"front": (1,H,W,3), "ego_view": ...}, "text": str} -> {"action": (horizon, D)}).

ROBOT x RETARGET -> which policy the bridge must be serving
    dex3    none             Stage A: Dex3 policy, 28-D end to end
    brainco dex3_to_brainco  Stage B: Dex3 (Stage A) policy driving the BrainCo robot.
                             26-D brainco state -> inverse_retarget_hand -> 28-D state;
                             28-D action chunk -> retarget_hand on the hand columns -> 26-D.
    brainco none             Stage C: BrainCo policy, 26-D end to end.
--dim is inferred from the robot (28 / 26); the bridge dimension is checked on the first
chunk. Scene, cameras, recorder and staged metrics are shared with handover_demos.py
(scripts/handover_common.py), so Stage A/B/C numbers are computed by the same code.

--configs <json>  list of {"object": .., "pose_noise": .., "seed": .., "held_out": bool}
                  runs every entry in ONE process (all three object kinds are spawned, the
                  inactive ones are parked away from the table) and aggregates M1-M6 into
                  results_<stage>.json + results_<stage>.md.

OUTPUT  <out>/<stage>_<object>_pn<noise>_s<seed>_ep<i>.json, ..._states.npy, ..._actions.npy,
        ..._tips.npy, cam/{front,ego_view}/<same stem>.mp4, results_<stage>.{json,md}
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from handover_common_args import add_common_args  # noqa: E402  (pure argparse)

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=None, help="default results/rollout_<robot>_<stage>")
parser.add_argument("--stage-label", type=str, default=None, choices=["A", "B", "C"],
                    help="default: A for dex3, B for brainco+retarget, C for brainco")
parser.add_argument("--retarget", type=str, default=None, choices=["none", "dex3_to_brainco"],
                    help="default: dex3_to_brainco when --robot brainco, else none")
parser.add_argument("--dim", type=int, default=None, help="robot state/action dim; inferred (28/26)")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=5771)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=750)
parser.add_argument("--execution-horizon", type=int, default=16)
parser.add_argument("--stop-after-place", type=int, default=60,
                    help="control steps to keep simulating after the place is detected (0 = run to max)")
parser.add_argument("--task-desc", type=str, default=None)
parser.add_argument("--configs", type=str, default=None,
                    help="json list of {object, pose_noise, seed, held_out}; overrides --object/--pose-noise/--seed")
add_common_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True          # the policy needs the cameras even with --no-video
if args_cli.retarget is None:
    args_cli.retarget = "dex3_to_brainco" if args_cli.robot == "brainco" else "none"
if args_cli.robot == "dex3" and args_cli.retarget != "none":
    raise SystemExit("--retarget dex3_to_brainco only makes sense with --robot brainco")
if args_cli.stage_label is None:
    args_cli.stage_label = ("A" if args_cli.robot == "dex3"
                            else "B" if args_cli.retarget == "dex3_to_brainco" else "C")
if args_cli.out is None:
    args_cli.out = f"results/rollout_{args_cli.robot}_{args_cli.stage_label}"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import pickle  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

import handover_common as hc  # noqa: E402

ROBOT_DIM = hc.STATE_DIM[args_cli.robot]
if args_cli.dim is None:
    args_cli.dim = ROBOT_DIM
elif args_cli.dim != ROBOT_DIM:
    raise SystemExit(f"--dim {args_cli.dim} does not match {args_cli.robot} ({ROBOT_DIM})")
POLICY_DIM = 28 if args_cli.retarget == "dex3_to_brainco" else ROBOT_DIM
TASK_TEXT = args_cli.task_desc or hc.TASK_TEXT


class Bridge:
    """Length-prefixed pickle RPC, identical to dex3_slot_rollout.py."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=300)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _recvn(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise ConnectionError("bridge closed")
            buf.extend(c)
        return bytes(buf)

    def _rpc(self, obj):
        p = pickle.dumps(obj, protocol=4)
        self.sock.sendall(struct.pack(">Q", len(p)) + p)
        n = struct.unpack(">Q", self._recvn(8))[0]
        return pickle.loads(self._recvn(n))

    def ping(self):
        return self._rpc({"op": "ping"})

    def act(self, state, video, text):
        r = self._rpc({"op": "act", "state": state, "video": video, "text": text})
        if "error" in r:
            raise RuntimeError("policy bridge raised:\n" + r["error"])
        return np.asarray(r["action"], dtype=np.float32)

    def close(self):
        try:
            self._rpc({"op": "bye"})
        except OSError:
            pass
        self.sock.close()


def load_configs():
    if args_cli.configs:
        with open(args_cli.configs) as f:
            cfgs = json.load(f)
        out = []
        for c in cfgs:
            if c.get("object", args_cli.object) not in hc.OBJECT_KINDS:
                raise SystemExit(f"unknown object in --configs: {c}")
            out.append({"object": c.get("object", args_cli.object),
                        "pose_noise": float(c.get("pose_noise", args_cli.pose_noise)),
                        "seed": int(c.get("seed", args_cli.seed)),
                        "held_out": bool(c.get("held_out", False)),
                        "episodes": int(c.get("episodes", args_cli.episodes))})
        return out
    return [{"object": args_cli.object, "pose_noise": args_cli.pose_noise, "seed": args_cli.seed,
             "held_out": False, "episodes": args_cli.episodes}]


def cfg_stem(cfg):
    return f"{args_cli.stage_label}_{cfg['object']}_pn{cfg['pose_noise']:g}_s{cfg['seed']}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def aggregate(cfg, recs):
    n = len(recs)
    ok = sum(r["success"] for r in recs)
    fails = {}
    for r in recs:
        fails[r["failure"]] = fails.get(r["failure"], 0) + 1
    return {
        **{k: cfg[k] for k in ("object", "pose_noise", "seed", "held_out")},
        "trials": n, "successes": ok, "success_rate": ok / max(1, n),
        "grasp_rate": sum(r["t_grasp"] is not None for r in recs) / max(1, n),
        "transfer_rate": sum(r["t_transfer"] is not None for r in recs) / max(1, n),
        "wrist_err_mean": _mean([(r["wrist_err"] or {}).get("mean") for r in recs]),
        "fingertip_err_mean": _mean([(r["fingertip_err"] or {}).get("mean_primary") for r in recs]),
        "drops": sum(r["dropped"] for r in recs), "falls": sum(r["fell"] for r in recs),
        "contact_spikes": sum(r["contact_spikes"] for r in recs),
        "joint_limit_violations": sum(r["joint_limit_violations"] for r in recs),
        "completion_step_mean": _mean([r["completion_step"] for r in recs]),
        "completion_s_mean": _mean([r["completion_s"] for r in recs]),
        "control_hz_nominal": 120.0 / args_cli.decimation,
        "control_hz_achieved_mean": _mean([r["control_hz_achieved"] for r in recs]),
        "failures": fails,
    }


def write_results(per_cfg, all_recs, bridge_info):
    stage = args_cli.stage_label
    overall = aggregate({"object": "all", "pose_noise": None, "seed": None, "held_out": None}, all_recs)
    fails = {}
    for r in all_recs:
        fails[r["failure"]] = fails.get(r["failure"], 0) + 1
    dominant = max(fails.items(), key=lambda kv: kv[1])[0] if fails else None
    summary = {
        "stage": stage, "robot": args_cli.robot, "retarget": args_cli.retarget,
        "robot_dim": ROBOT_DIM, "policy_dim": POLICY_DIM, "task_text": TASK_TEXT,
        "execution_horizon": args_cli.execution_horizon, "max_steps": args_cli.max_steps,
        "control_hz_nominal": 120.0 / args_cli.decimation, "bridge": bridge_info,
        "per_config": per_cfg, "overall": overall,
        "failure_categories": fails, "dominant_failure": dominant,
        "M5_note": "adaptation cost is a training-side quantity; see the finetune logs",
        "episodes": all_recs,
    }
    with open(os.path.join(args_cli.out, f"results_{stage}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    hdr = ("| config | held out | trials | success | grasp | transfer | wrist err (m) | "
           "fingertip err (m) | drops | falls | contacts | limit viol | completion step | "
           "completion (s) | Hz (nominal / achieved) |")
    lines = [f"# Stage {stage} results ({args_cli.robot}, retarget={args_cli.retarget}, "
             f"policy dim {POLICY_DIM})", "", hdr, "|" + "---|" * 15]
    for a in per_cfg + [overall]:
        name = ("overall" if a["object"] == "all"
                else f"{a['object']} pn{a['pose_noise']:g} s{a['seed']}")
        lines.append("| " + " | ".join([
            name, "-" if a["held_out"] is None else ("yes" if a["held_out"] else "no"),
            str(a["trials"]), f"{a['successes']}/{a['trials']} ({100 * a['success_rate']:.0f}%)",
            f"{100 * a['grasp_rate']:.0f}%", f"{100 * a['transfer_rate']:.0f}%",
            _fmt(a["wrist_err_mean"]), _fmt(a["fingertip_err_mean"]),
            str(a["drops"]), str(a["falls"]), str(a["contact_spikes"]),
            str(a["joint_limit_violations"]), _fmt(a["completion_step_mean"], 1),
            _fmt(a["completion_s_mean"], 2),
            f"{a['control_hz_nominal']:.0f} / {_fmt(a['control_hz_achieved_mean'], 1)}"]) + " |")
    lines += ["", f"Dominant failure mode: {dominant} ({fails})",
              "", "contacts = control steps where the object's linear velocity jumped by more than "
              f"{args_cli.vel_spike} m/s (proxy for unintended table/robot contact). "
              "wrist / fingertip errors are vs the Dex3 reference run given by --ref-tips-dir "
              "(n/a without it). Completion time at the nominal control rate."]
    with open(os.path.join(args_cli.out, f"results_{stage}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    configs = load_configs()
    bridge = Bridge(args_cli.host, args_cli.port)
    pong = bridge.ping()
    print(f"HROLL_PING {pong}", flush=True)

    scene_cfg = hc.make_scene_cfg(args_cli)
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args_cli.device))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    hc.set_stick_com(scene, args_cli)
    dt = sim.get_physics_dt()
    dev = sim.device
    robot = scene["robot"]
    names = list(robot.data.joint_names)
    hand = hc.HandModel(robot, args_cli.robot, args_cli.grip)
    bodies = list(robot.data.body_names)
    wr_R, wr_L = bodies.index("right_wrist_yaw_link"), bodies.index("left_wrist_yaw_link")
    print(f"HROLL_ROBOT {args_cli.robot} stage={args_cli.stage_label} retarget={args_cli.retarget} "
          f"robot_dim={ROBOT_DIM} policy_dim={POLICY_DIM}", flush=True)
    print(f"HROLL_JOINTS {len(names)} {names}", flush=True)
    print(f"HROLL_HAND {hand.hand_names}", flush=True)
    if args_cli.robot == "brainco" and not hc.mimic_ok(hand):
        raise SystemExit(f"expected 10 mimic pairs, got {len(hand.mimic_pairs)}")

    default = robot.data.default_joint_pos.clone()
    default_vel = torch.zeros_like(robot.data.default_joint_vel)
    root_z_init = float(robot.data.root_pos_w[0, 2])
    tip_ref = hc.TipReference(args_cli.ref_tips_dir) if args_cli.ref_tips_dir else None
    n_phys = args_cli.decimation
    place = np.array(args_cli.place_xy, np.float64)
    ox, oy = args_cli.obj_start

    def settle(n):
        for _ in range(n):
            robot.set_joint_position_target(default)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

    hc.measure_table_top(scene, settle, dev)          # the constant TABLE_TOP_Z is 4 cm off

    per_cfg, all_recs = [], []
    for cfg in configs:
        obj = scene[f"obj_{cfg['object']}"]
        sz = hc.spawn_z(cfg["object"])          # measured table top (hc.TABLE_TOP)
        rng = np.random.default_rng(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        stem_cfg = cfg_stem(cfg)
        recs = []
        print(f"HROLL_CONFIG {stem_cfg} held_out={cfg['held_out']} episodes={cfg['episodes']}", flush=True)

        for ep in range(cfg["episodes"]):
            jx, jy = rng.uniform(-cfg["pose_noise"], cfg["pose_noise"], 2)
            robot.write_joint_state_to_sim(default.clone(), default_vel.clone())   # robot home FIRST, then the object
            robot.reset()
            settle(10)
            hc.place_objects(scene, cfg["object"], (ox + jx, oy + jy), sz, dev)
            settle(60)
            # Isaac Sim 5.1 on this machine sometimes starts a process whose tiled cameras never
            # render (all-zero frames). A blind policy result is not a result: warm up, re-check,
            # and exit with code 3 so the driver relaunches the config.
            def _black():
                return [kk for kk, _ in hc.CAM_KEYS
                        if float(scene[kk].data.output["rgb"][0, ..., :3].float().mean()) < 1.0]
            _bl = _black()
            for _ in range(30):
                if not _bl:
                    break
                sim.render(); scene.update(dt); _bl = _black()
            if _bl:
                print(f"HROLL_BLACK_CAMERA {_bl} config={stem_cfg} ep={ep}: exiting for relaunch", flush=True)
                os._exit(3)
            obj_p0 = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            z0 = float(obj_p0[2])
            _rt = {"rest_z": hc.rest_z_for(cfg["object"])}
            if getattr(args_cli, "stick", 0):       # same rule as the demos: any pose resting ON the table
                _top = hc.table_top_z(); _lo = _top + args_cli.stick_w / 2; _hi = _top + args_cli.stick_h / 2
                _rt = {"rest_z": 0.5 * (_lo + _hi), "rest_tol": 0.5 * (_hi - _lo) + 0.01}
            tracker = hc.EpisodeTracker(z0, place, root_z_init, args_cli.vel_spike, None,
                                        place_radius=getattr(args_cli, 'place_radius', 0.12), **_rt)

            targets = default.clone()
            chunk, k = None, 0
            states, actions, tips = [], [], []
            buf = {t: [] for _, t in hc.CAM_KEYS}
            n_queries, t_bridge = 0, 0.0
            step_after_place = None
            t0 = time.perf_counter()

            for t in range(args_cli.max_steps):
                if chunk is None or k >= min(args_cli.execution_horizon, len(chunk)):
                    q = robot.data.joint_pos[0].cpu().numpy()
                    s = hand.state(q).astype(np.float32)
                    if args_cli.retarget == "dex3_to_brainco":
                        hands = hand.hand_block_to_dex3(s[14:]).astype(np.float32)   # 12 -> 14
                    else:
                        hands = s[14:]
                    state = {"arms": s[:14][None, :], "hands": hands[None, :]}
                    video = {}
                    for key, tag in hc.CAM_KEYS:
                        fr = scene[key].data.output["rgb"][0, ..., :3]
                        video[tag] = fr.detach().cpu().numpy().astype(np.uint8)[None, ...]
                    tb = time.perf_counter()
                    chunk = bridge.act(state, video, TASK_TEXT)
                    t_bridge += time.perf_counter() - tb
                    n_queries += 1
                    if ep == 0 and t == 0 and cfg is configs[0]:
                        print(f"HROLL_CHUNK {chunk.shape}", flush=True)
                    if chunk.ndim != 2 or chunk.shape[1] != POLICY_DIM:
                        raise SystemExit(f"bridge returned {chunk.shape}, expected (H, {POLICY_DIM}); "
                                         f"is the Stage {args_cli.stage_label} policy loaded?")
                    k = 0
                a = chunk[k].astype(np.float64)
                k += 1
                hand.write_arms(targets, a[:14])
                if args_cli.retarget == "dex3_to_brainco":
                    hand.write_hand(targets, hand.hand_block_from_dex3(a[14:28]))     # 14 -> 12 (+mimic)
                else:
                    hand.write_hand(targets, a[14:])
                robot.set_joint_position_target(targets)
                for _ in range(n_phys):
                    scene.write_data_to_sim(); sim.step(); scene.update(dt)

                q = robot.data.joint_pos[0].cpu().numpy()
                states.append(hand.state(q))
                actions.append(hand.action_vec(targets[0].cpu().numpy()))
                tips.append(hc.tips_now(robot, hand))
                if not args_cli.no_video:
                    hc.grab_frames(scene, buf)
                op = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
                ov = obj.data.root_lin_vel_w[0].cpu().numpy().astype(np.float64)
                pR = robot.data.body_pos_w[0, wr_R].cpu().numpy().astype(np.float64)
                pL = robot.data.body_pos_w[0, wr_L].cpu().numpy().astype(np.float64)
                tracker.update(op, ov, pR, pL, float(robot.data.root_pos_w[0, 2]),
                               hand.limit_violation(q), log_prefix=f"{stem_cfg} ep{ep} ")
                if tracker.fell:
                    break
                if tracker.t_place is not None and args_cli.stop_after_place > 0:
                    step_after_place = (step_after_place or 0) + 1
                    if step_after_place >= args_cli.stop_after_place:
                        break
            wall = time.perf_counter() - t0
            n_steps = tracker.step
            if getattr(args_cli, "stick", 0):         # placed stick must stand/lie on its own, hands released
                tracker.check_free(obj.data.root_pos_w[0].cpu().numpy(), obj.data.root_quat_w[0].cpu().numpy(),
                                   hc.tips_now(robot, hand), 0.5 * args_cli.stick_h, 0.5 * args_cli.stick_w)
            tips_arr = np.array(tips, dtype=np.float32)
            stem = f"{stem_cfg}_ep{ep:03d}"
            rec = {"config": stem_cfg, "stage": args_cli.stage_label, "robot": args_cli.robot,
                   "object": cfg["object"], "pose_noise": cfg["pose_noise"], "seed": cfg["seed"],
                   "held_out": cfg["held_out"], "episode": ep,
                   "obj_start": obj_p0.round(4).tolist(),
                   "obj_end": obj.data.root_pos_w[0].cpu().numpy().round(4).tolist(), "z0": z0,
                   **tracker.record(), "failure": tracker.failure_category(),
                   "completion_s": (tracker.t_place * n_phys / 120.0 if tracker.t_place is not None else None),
                   "wrist_err": tip_ref.wrist_error(tips_arr, ep) if tip_ref else None,
                   "fingertip_err": tip_ref.error(tips_arr, hand, ep) if tip_ref else None,
                   "steps": n_steps, "wall_s": wall, "control_hz_achieved": n_steps / max(wall, 1e-6),
                   "bridge_queries": n_queries, "bridge_s_per_query": t_bridge / max(1, n_queries),
                   "files": {"states": f"{stem}_states.npy", "actions": f"{stem}_actions.npy",
                             "tips": f"{stem}_tips.npy"}}
            np.save(os.path.join(args_cli.out, f"{stem}_states.npy"), np.array(states, dtype=np.float32))
            np.save(os.path.join(args_cli.out, f"{stem}_actions.npy"), np.array(actions, dtype=np.float32))
            np.save(os.path.join(args_cli.out, f"{stem}_tips.npy"), tips_arr)
            if not args_cli.no_video:
                rec["videos"] = hc.write_videos(buf, args_cli.out, stem, args_cli.video_fps)
            rec["jl_joints"] = hand.jl_report()     # per-joint steps past the hard limit (resets per episode)
            with open(os.path.join(args_cli.out, f"{stem}.json"), "w") as f:
                json.dump(rec, f, indent=2)
            recs.append(rec); all_recs.append(rec)
            print(f"HROLL_EP {stem} grasp={tracker.t_grasp} transfer={tracker.t_transfer} "
                  f"place={tracker.t_place} dropped={tracker.dropped} fell={tracker.fell} "
                  f"jl={tracker.jl_viol} spikes={tracker.contact_spikes} success={tracker.success} "
                  f"steps={n_steps} hz={rec['control_hz_achieved']:.1f} jl_joints={rec['jl_joints']}", flush=True)
        agg = aggregate(cfg, recs)
        per_cfg.append(agg)
        print(f"HROLL_CONFIG_DONE {stem_cfg} success={agg['successes']}/{agg['trials']}", flush=True)
        # keep partial results on disk in case a later config dies
        write_results(per_cfg, all_recs, {"pong": str(pong), "host": args_cli.host, "port": args_cli.port})

    write_results(per_cfg, all_recs, {"pong": str(pong), "host": args_cli.host, "port": args_cli.port})
    n_ok = sum(r["success"] for r in all_recs)
    print(f"HROLL_SUCCESS_RATE {n_ok}/{len(all_recs)}", flush=True)
    bridge.close()
    print("HROLL_OK", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # Kit can hang in close(); results are already written
