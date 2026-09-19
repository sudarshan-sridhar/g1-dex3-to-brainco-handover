"""Convert the official Unitree G1 + BrainCo Revo2 URDF to USD with Isaac Lab's UrdfConverter.

Input : assets_brainco/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
Output: assets_brainco/usd/g1_29dof_brainco.usd  (+ configuration/ layers written by the importer)

Before conversion a temp copy of the URDF is written next to the original (so relative
``meshes/...`` paths still resolve) with two fixes, see docs/hand_models.md:
  * any ``package://`` mesh path is rewritten to an absolute path (none in the official file,
    kept for the unofficial variant);
  * the 10 degenerate ``*_tip`` revolute joints (lower == upper) are turned into fixed joints
    unless ``--keep-tip-joints`` is given.

Mimic joints: by default ``convert_mimic_joints_to_normal_joints=True`` (distal joints become
plain drives; couple them in software with ``brainco_cfg.BRAINCO_MIMIC``). ``--keep-mimic``
leaves PhysX mimic joints in place instead.

Usage (from the Isaac Lab python env, headless):
    python scripts/convert_brainco_urdf.py --headless
    python scripts/convert_brainco_urdf.py --headless --input <other.urdf> --output <out.usd>

API reference: IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py,
               IsaacLab/scripts/tools/convert_urdf.py
"""

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
DEFAULT_URDF = os.path.join(
    _PROJECT, "assets_brainco", "unitree_ros", "robots", "g1_with_brainco_hand",
    "g1_29dof_mode_15_brainco_hand.urdf",
)
DEFAULT_USD = os.path.join(_PROJECT, "assets_brainco", "usd", "g1_29dof_brainco.usd")

parser = argparse.ArgumentParser(description="URDF -> USD for G1 + BrainCo Revo2.")
parser.add_argument("--input", default=DEFAULT_URDF, help="Input URDF path.")
parser.add_argument("--output", default=DEFAULT_USD, help="Output USD path.")
parser.add_argument("--keep-mimic", action="store_true",
                    help="Keep <mimic> as PhysX mimic joints instead of converting to normal joints.")
parser.add_argument("--keep-tip-joints", action="store_true",
                    help="Do not rewrite degenerate *_tip revolute joints to fixed.")
parser.add_argument("--self-collision", action="store_true", help="Enable self collisions.")
parser.add_argument("--collider", default="convex_hull", choices=["convex_hull", "convex_decomposition"])
# Body drive gains written into the USD. Isaac Lab actuator cfgs override these at spawn,
# so they only matter when the USD is opened outside an ArticulationCfg.
parser.add_argument("--joint-stiffness", type=float, default=100.0)
parser.add_argument("--joint-damping", type=float, default=1.0)
parser.add_argument("--hand-stiffness", type=float, default=10.0)
parser.add_argument("--hand-damping", type=float, default=0.5)
parser.add_argument("--keep-temp", action="store_true", help="Leave the patched temp URDF on disk.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Pure conversion; force headless regardless of the flag so no viewport is created.
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

HAND_JOINT_RE = re.compile(r"^(left|right)_(thumb|index|middle|ring|pinky)_(metacarpal|proximal|distal)_joint$")


def patch_urdf(src: str, fix_tips: bool) -> tuple[str, dict]:
    """Write a patched copy of ``src`` in the same directory; return (path, stats)."""
    src_dir = os.path.dirname(os.path.abspath(src))
    tree = ET.parse(src)
    root = tree.getroot()
    stats = {"package_rewritten": 0, "tips_fixed": 0, "missing_meshes": []}

    # 1) mesh paths: package:// -> absolute; relative paths are left alone because the temp
    #    copy lives in the same directory as the original.
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            rel = fn[len("package://"):].split("/", 1)[1] if "/" in fn[len("package://"):] else ""
            cand = os.path.join(src_dir, rel)
            if not os.path.exists(cand):
                # package root may be the parent of the URDF dir (ros package layout)
                cand = os.path.join(os.path.dirname(src_dir), rel)
            mesh.set("filename", os.path.abspath(cand))
            stats["package_rewritten"] += 1
        resolved = mesh.get("filename")
        if not os.path.isabs(resolved):
            resolved = os.path.join(src_dir, resolved)
        if not os.path.exists(resolved):
            stats["missing_meshes"].append(mesh.get("filename"))

    # 2) degenerate tip joints (lower == upper) -> fixed
    if fix_tips:
        for joint in root.findall("joint"):
            if joint.get("type") != "revolute":
                continue
            lim = joint.find("limit")
            if lim is None:
                continue
            if float(lim.get("lower", 0)) == float(lim.get("upper", 0)):
                joint.set("type", "fixed")
                for tag in ("axis", "limit", "mimic", "dynamics"):
                    el = joint.find(tag)
                    if el is not None:
                        joint.remove(el)
                stats["tips_fixed"] += 1

    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(src_dir, f"_{base}_isaac_tmp.urdf")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out, stats


def main():
    urdf_in = os.path.abspath(args_cli.input)
    usd_out = os.path.abspath(args_cli.output)
    if not os.path.isfile(urdf_in):
        raise FileNotFoundError(urdf_in)
    os.makedirs(os.path.dirname(usd_out), exist_ok=True)

    patched, stats = patch_urdf(urdf_in, fix_tips=not args_cli.keep_tip_joints)
    print(f"BRAINCO_PATCHED {patched}")
    print(f"BRAINCO_PATCH_STATS {stats}")
    if stats["missing_meshes"]:
        raise FileNotFoundError(f"missing meshes: {stats['missing_meshes']}")

    # Per-joint gains: body joints get the generic values, hand joints the lighter ones
    # (same numbers as brainco_cfg.G1_BRAINCO_CFG so the bare USD behaves like the cfg).
    joint_names = [j.get("name") for j in ET.parse(patched).getroot().findall("joint")
                   if j.get("type") != "fixed"]
    stiffness = {n: (args_cli.hand_stiffness if HAND_JOINT_RE.match(n) else args_cli.joint_stiffness)
                 for n in joint_names}
    damping = {n: (args_cli.hand_damping if HAND_JOINT_RE.match(n) else args_cli.joint_damping)
               for n in joint_names}

    cfg = UrdfConverterCfg(
        asset_path=patched,
        usd_dir=os.path.dirname(usd_out),
        usd_file_name=os.path.basename(usd_out),
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=True,
        root_link_name="pelvis",
        link_density=0.0,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=not args_cli.keep_mimic,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping),
        ),
        collider_type=args_cli.collider,
        self_collision=args_cli.self_collision,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
    )
    print("-" * 80)
    print_dict({k: v for k, v in cfg.to_dict().items() if k != "joint_drive"}, nesting=0)
    print(f"joint_drive: {len(joint_names)} joints, hand stiffness/damping "
          f"{args_cli.hand_stiffness}/{args_cli.hand_damping}, body "
          f"{args_cli.joint_stiffness}/{args_cli.joint_damping}")
    print("-" * 80)

    converter = UrdfConverter(cfg)
    print(f"BRAINCO_USD {converter.usd_path}")

    if not args_cli.keep_temp:
        os.remove(patched)
    else:
        print(f"BRAINCO_TEMP_KEPT {patched}")

    # Quick sanity: count joints in the generated stage.
    try:
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(converter.usd_path)
        joints = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        rev = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        print(f"BRAINCO_USD_JOINTS total={len(joints)} revolute={len(rev)}")
        hand = [n for n in rev if HAND_JOINT_RE.match(n)]
        print(f"BRAINCO_USD_HAND_JOINTS {len(hand)} (expect 22)")
    except Exception as e:  # pragma: no cover
        print(f"BRAINCO_USD_INSPECT_SKIPPED {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()
