"""Isaac Lab ArticulationCfg for the Unitree G1 29-DOF with BrainCo Revo2 hands.

Modelled on ``G1_29DOF_CFG`` (IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:388).
The USD is produced by ``scripts/convert_brainco_urdf.py`` from the official
``assets_brainco/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf``.
Joint facts are documented in ``docs/hand_models.md``.

Hand kinematics (per side, 16 revolute joints in the URDF, 6 real DOF):
  thumb : metacarpal (z) -> proximal (x) -> distal (x, mimic proximal x1.0)
  index/middle/ring/pinky : proximal (y) -> distal (y, mimic proximal x1.155)
  plus one degenerate *_tip_joint per finger that the converter turns into a fixed joint.

The converter defaults to ``convert_mimic_joints_to_normal_joints=True``, so the 10 distal
joints are ordinary drives in the USD and the coupling must be applied in software with
``BRAINCO_MIMIC`` (see ``apply_mimic``).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

# Placeholder output of scripts/convert_brainco_urdf.py. Override with BRAINCO_USD env var.
BRAINCO_USD_PATH = os.environ.get(
    "BRAINCO_USD",
    os.path.join(_PROJECT, "assets_brainco", "usd", "g1_29dof_brainco.usd"),
)

# The 12 motor-backed joints, robot order (left hand then right hand), matching the
# order they appear in the official URDF within each hand.
BRAINCO_ACTUATED = [
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]

# distal joint -> (proximal joint it follows, multiplier). Offset is 0 for every pair.
BRAINCO_MIMIC = {
    f"{side}_{finger}_distal_joint": (f"{side}_{finger}_proximal_joint", ratio)
    for side in ("left", "right")
    for finger, ratio in (
        ("thumb", 1.0),
        ("index", 1.155),
        ("middle", 1.155),
        ("ring", 1.155),
        ("pinky", 1.155),
    )
}

# URDF joint limits (upper; lower is 0 everywhere) for clipping software-mimic targets.
BRAINCO_UPPER = {
    "thumb_metacarpal": 1.5184,
    "thumb_proximal": 1.0472,
    "thumb_distal": 1.0472,
    "index_proximal": 1.4661, "index_distal": 1.693,
    "middle_proximal": 1.4661, "middle_distal": 1.693,
    "ring_proximal": 1.4661, "ring_distal": 1.693,
    "pinky_proximal": 1.4661, "pinky_distal": 1.693,
}

# Regex that matches exactly the 22 drivable hand joints (12 actuated + 10 distal) and
# nothing else: tip joints are fixed after conversion, base joints are fixed in the URDF.
BRAINCO_HAND_JOINT_EXPR = [
    ".*_thumb_metacarpal_joint",
    ".*_thumb_proximal_joint",
    ".*_thumb_distal_joint",
    ".*_(index|middle|ring|pinky)_proximal_joint",
    ".*_(index|middle|ring|pinky)_distal_joint",
]


def apply_mimic(targets: dict[str, float]) -> dict[str, float]:
    """Fill in distal joint targets from their proximal targets (in place, returned)."""
    for distal, (proximal, ratio) in BRAINCO_MIMIC.items():
        if proximal in targets:
            key = distal.split("_", 1)[1].rsplit("_joint", 1)[0]  # e.g. index_distal
            targets[distal] = min(max(targets[proximal] * ratio, 0.0), 0.9 * BRAINCO_UPPER[key])  # soft limit
    return targets


G1_BRAINCO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BRAINCO_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Tabletop manipulation: pin the pelvis.
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # Same init pose as G1_29DOF_CFG.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ---- legs / feet / waist: verbatim copy of G1_29DOF_CFG ----
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 5000.0,
                "waist_roll_joint": 5000.0,
                "waist_pitch_joint": 5000.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.001,
        ),
        # ---- arms: verbatim copy of G1_29DOF_CFG ----
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=3000.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        # ---- BrainCo hands ----
        # Reasoning for the gains. The Dex3 group in G1_29DOF_CFG uses stiffness 20, damping 2,
        # effort 300, velocity 100 on 7 joints per hand whose links weigh tens of grams and
        # whose URDF efforts are a few Nm. Revo2 finger links are 9 g (proximal) and the URDF
        # rates the motors at 0.5 Nm (thumb metacarpal), 1.1 Nm (thumb) and 2 Nm (fingers),
        # velocity 2.3 to 2.6 rad/s: roughly half the Dex3 torque and a tenth the inertia.
        # Halving the Dex3 stiffness (20 -> 10) keeps the drive from exceeding the motor
        # rating over the full 1.5 rad travel (10 * 1.5 = 15 Nm demand, clipped to the
        # effort limit), and damping 0.5 keeps the ratio damping/sqrt(stiffness) similar
        # to Dex3 (2/sqrt(20) = 0.45 vs 0.5/sqrt(10) = 0.16, slightly under-damped on
        # purpose so the light links do not lag the proximal target). This also matches the
        # dexterous-hand gains (10 / 0.2) Isaac Lab ships for similar manipulation tasks.
        # Effort limits are 2x the URDF motor rating so contact can be held against the
        # 0.05 kg wheel without the drive saturating in the first frame; velocity is 2x the
        # URDF rating. Both are sim caps, not physical claims.
        "brainco_hands": ImplicitActuatorCfg(
            joint_names_expr=BRAINCO_HAND_JOINT_EXPR,
            effort_limit_sim={
                ".*_thumb_metacarpal_joint": 1.0,
                ".*_thumb_proximal_joint": 2.2,
                ".*_thumb_distal_joint": 2.2,
                ".*_(index|middle|ring|pinky)_proximal_joint": 4.0,
                ".*_(index|middle|ring|pinky)_distal_joint": 4.0,
            },
            velocity_limit_sim=5.0,
            stiffness=10.0,
            damping=0.5,
            armature=0.001,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)
"""G1 29-DOF + BrainCo Revo2 hands, fixed base, for tabletop manipulation replay/rollout."""
