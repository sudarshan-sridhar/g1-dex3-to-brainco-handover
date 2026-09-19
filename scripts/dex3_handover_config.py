"""GR00T N1.7 modality config for the Unitree G1 + Dex3 HANDOVER slot (Stage A).

Identical to dex3_config.py (28-D: arms 0:14, hands 14:28, ABSOLUTE joint space,
video ego_view + front, 16-step action horizon). Kept as a separate file so the
handover checkpoint and the earlier pick checkpoint never share a registration by
accident and each sbatch names the config it was actually trained with.

Notes carried over from dex3_config.py:

JOINT SPACE, DELIBERATELY
    The Dex3 USD cannot be converted to URDF (mixed parent conventions, closed-linkage
    fingers), so nothing downstream may depend on IK. Joints are commanded directly.

ONE GROUP PER BLOCK, NOT PER SIDE
    Both blocks are INTERLEAVED left/right in the robot's own joint ordering. There is
    no contiguous "left_arm" slice to declare; inventing one silently mixes the sides.

ABSOLUTE
    The recorder commands absolute joint targets and the rollout applies them directly.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

_ABS = ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT)

dex3_handover_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view", "front"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["arms", "hands"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["arms", "hands"],
        action_configs=[_ABS, _ABS],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(dex3_handover_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
