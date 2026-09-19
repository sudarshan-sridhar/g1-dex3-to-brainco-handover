"""Shared CLI flags for handover_demos.py / handover_rollout.py.

Pure argparse, no isaaclab import, so it can be called BEFORE AppLauncher parses.
"""

OBJECT_KINDS = ("steering_wheel", "cube", "cylinder", "bar")
ROBOTS = ("dex3", "brainco")

# --obj-start default per object. The mirrored joint-space grasp replay
# (--grasp-replay) is open loop and very sensitive to the wheel pose: 0.02 m moves
# the hand onto a different part of the rim, so the wheel MUST spawn where
# the replay was tuned. The cylinder/cube pose is the run-7 IK-grasp pose.
OBJ_START_DEFAULT = {"steering_wheel": (0.208, 0.295),
                     "cube": (0.05, 0.38),
                     "cylinder": (0.05, 0.38),
                     # bar: horizontal, axis along world x (the robot's left-right), resting
                     # on two risers, centred in front of the chest and 0.06 m clear of the
                     # home right fingertips (which reach y=0.366).
                     "bar": (0.0, 0.45)}

# per-object default mass (kg): the 0.30 x 0.02 bar is 0.20 kg (coordinator decision), the
# rest keep the historical 0.05 kg.
OBJ_MASS_DEFAULT = {"bar": 0.20}


def resolve_obj_mass(args):
    if getattr(args, "obj_mass", None) is None:
        args.obj_mass = OBJ_MASS_DEFAULT.get(args.object, 0.05)
    return float(args.obj_mass)


def resolve_obj_start(args):
    """Fill in args.obj_start from OBJ_START_DEFAULT when it was not given."""
    if getattr(args, "obj_start", None) is None:
        args.obj_start = OBJ_START_DEFAULT[args.object]
    return tuple(args.obj_start)


# ---------------------------------------------------------------------------
def add_common_args(parser):
    parser.add_argument("--stick", type=int, default=0, help="1: --object cylinder becomes an upright square stick")
    parser.add_argument("--bc-oppose", type=float, nargs="+", default=None, help="BrainCo-native opposition grasp: M_PRE P_CLOSE F_CLOSE [T0 T1 F0 F1]. Thumb metacarpal held at M_PRE (1.5184 = fully across the palm), thumb proximal closes to P_CLOSE over ramp window [T0,T1], fingers to F_CLOSE over [F0,F1] (defaults 0.3 1.0 0.0 0.7)")
    parser.add_argument("--thumb-lead", type=float, default=0.0, help="BrainCo: fraction of each close ramp in which the thumb closes before the fingers start (0 = together)")
    parser.add_argument("--stick-com", type=float, default=None, help="stick: centre-of-mass offset along the stick axis from its centre (e.g. -0.08 = weighted base)")
    parser.add_argument("--stick-round", type=int, default=0, help="1: round stick (held-out object)")
    parser.add_argument("--stick-collar", type=float, default=0.0, help="stick: height of a static square collar around the stick base (0 = none). Feasibility test for the BrainCo grasp: the stick cannot tip while a hand closes; lifting pulls it out")
    parser.add_argument("--stick-w", type=float, default=0.03)
    parser.add_argument("--stick-h", type=float, default=0.24)
    parser.add_argument("--bar-shape", type=str, default="cyl", choices=["cyl", "box"], help="bar cross-section: cyl (rolls) or box (square, cannot roll)")
    parser.add_argument("--bar-axis", type=str, default="x", choices=["x", "y"], help="bar long axis: x (left-right) or y (away from the robot)")

    parser.add_argument("--robot", type=str, default="dex3", choices=list(ROBOTS))
    parser.add_argument("--object", type=str, default="cylinder", choices=list(OBJECT_KINDS))
    parser.add_argument("--grip", type=float, default=0.9,
                        help="fraction of each Dex3 finger joint's soft travel when closed "
                             "(also the closure reference for BrainCo retargeting)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pose-noise", type=float, default=0.03,
                        help="uniform +-xy jitter (m) on the object start pose")
    parser.add_argument("--decimation", type=int, default=6,
                        help="physics steps (dt 1/120) per control step -> 20 Hz")
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--cam-res", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--table-z", type=float, default=1.0)
    parser.add_argument("--obj-start", type=float, nargs=2, default=None,
                        help="object start xy; default depends on --object (OBJ_START_DEFAULT): "
                             "the steering wheel keeps the pose the mirrored grasp replay was "
                             "tuned against, the cylinder/cube sit 0.10 m clear of the home "
                             "right fingertips. Recorded in handover_summary.json.")
    parser.add_argument("--place-xy", type=float, nargs=2, default=(-0.05, 0.38))
    parser.add_argument("--front-eye", type=float, nargs=3, default=(0.75, 1.15, 1.60))
    parser.add_argument("--front-target", type=float, nargs=3, default=(0.0, 0.28, 1.10))
    parser.add_argument("--ego-eye", type=float, nargs=3, default=(0.0, 0.10, 1.50))
    parser.add_argument("--ego-target", type=float, nargs=3, default=(0.0, 0.30, 1.06))
    parser.add_argument("--ref-tips-dir", type=str, default=None,
                        help="directory with ep{i}_tips.npy + handover_summary.json from a "
                             "Dex3 run; enables the M2 fingertip tracking error")
    parser.add_argument("--vel-spike", type=float, default=0.6,
                        help="object |dv| per control step (m/s) counted as an unintended contact")
    return parser
