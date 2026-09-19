"""GR00T N1.7 modality config for the Unitree G1 + BrainCo Revo2 embodiment (Stage C).

Sibling of dex3_config.py; only the hand width differs:

    Dex3     28-D action  arms(14) hands(14)   JOINT-space, ABSOLUTE
    BrainCo  26-D action  arms(14) hands(12)   JOINT-space, ABSOLUTE

The 12 hand columns are brainco_cfg.BRAINCO_ACTUATED in order (left thumb_metacarpal,
thumb_proximal, index/middle/ring/pinky proximal, then the same six on the right).
The 10 distal joints are mimic-coupled in software (brainco_cfg.apply_mimic) and are
NOT part of the action or state, so the policy never sees them.

The arms block is the same interleaved-left/right 14 joints as Dex3, so a checkpoint
finetuned on the Dex3 handover slot carries its arm statistics straight across; only
the hands sub-block changes width. Video keys (ego_view, front) and the language key
are identical to dex3_config.py so the Stage A checkpoint's vision/language path is
reused unchanged (requirement 1.5: tune projector + diffusion head only).
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

# Widths are fixed by the dataset's meta/modality.json (arms 0:14, hands 14:26); the
# modality config only names the groups and their order.
brainco_config = {
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

register_modality_config(brainco_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
