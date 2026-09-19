# Source policy, versions, licenses and modifications

## Starting checkpoint

| | |
|:--|:--|
| model | `nvidia/GR00T-N1.7-3B` on Hugging Face |
| revision | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| size | 6.4 GB, 1031 tensors, includes its own Eagle vision-language backbone |
| license | NVIDIA Open Model License (LICENSE file in the snapshot) |
| code | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`, Apache-2.0 |
| runtime | Python 3.12, torch 2.9.0+cu128, attention through PyTorch SDPA (the flash-attention wheel needs a newer glibc than the cluster has) |

**Original embodiment.** The checkpoint's G1 entry (`real_g1_relative_eef_relative_joints`) uses,
per side, a 9-value wrist end-effector pose, 7 arm joints and 7 hand joints, which is a G1 with
Dex3 hands, plus a 3-value waist; the action adds a base height command and a 3-value navigation
command. NVIDIA's own comment on the hand entries is that they are "controlled by binary signals
like a gripper". Other pretrained embodiments in the same checkpoint cover xDoF arms, OXE DROID,
and the R1 Pro with Sharpa hands.

**Observation and action spaces used here.** Two 224x224 RGB cameras (head and front), joint
positions, and a fixed task sentence. Actions are absolute joint position targets, 16 steps per
inference: 28 values with Dex3 hands (14 arm, 14 hand), 26 with Brainco hands (14 arm, 12 hand).

**Control frequency.** 20 Hz here (120 Hz physics, decimation 6). The pretraining control rate is
not recorded in the checkpoint metadata.

**Task distribution (model card).** A cross-embodiment mix of real robot teleoperation, open
datasets such as OXE DROID and LIBERO, and simulation including tabletop manipulation. Exact
proportions are not published.

## Material modifications

The base weights are never edited. Everything below sits outside them.

1. A `NEW_EMBODIMENT` slot trained for each hand: 28 values for the Dex3 layout, 26 for Brainco.
2. Modality configs registered at load time: `scripts/dex3_handover_config.py` and
   `scripts/brainco_config.py`. Both declare two cameras, the state and action groups, absolute
   joint targets and an action horizon of 16.
3. Dataset statistics regenerated per dataset, as the action width and hand layout change.
4. A second camera in the simulation rig. The stock G1 configuration renders one.
5. Fine-tuning settings: learning rate 1e-4, batch 16, state dropout 0.05, vision-language
   backbone frozen, projector and flow-matching action head trained.
6. No patches to the GR00T source.

## Exported checkpoints

Both are public and contain the weights, the modality configuration, the dataset statistics and a
provenance file. Optimizer state is not included.

| stage | model | training |
|:--|:--|:--|
| A, Dex3 | [Sudarshan18/gr00t-n17-g1-dex3-handover](https://huggingface.co/Sudarshan18/gr00t-n17-g1-dex3-handover) | 98 demonstrations, 66,745 frames, 16,000 steps, 3 h 29 min on one A40 |
| C, Brainco Revo2 | [Sudarshan18/gr00t-n17-g1-brainco-handover](https://huggingface.co/Sudarshan18/gr00t-n17-g1-brainco-handover) | starts from Stage A; 44 demonstrations, 41,360 frames, 10,000 steps, 2 h 12 min on one A40 |

Machine-readable records: [../results/checkpoints/stage_a_provenance.json](../results/checkpoints/stage_a_provenance.json)
and [../results/checkpoints/stage_c_provenance.json](../results/checkpoints/stage_c_provenance.json).
Loss histories: [../results/training](../results/training).

## External components

| component | version | license | why |
|:--|:--|:--|:--|
| NVIDIA Isaac-GR00T | commit `51d4c89` | Apache-2.0 | policy, fine-tuning and inference |
| GR00T N1.7-3B weights | revision `2fc962b` | NVIDIA Open Model License | the pretrained policy the assignment asks to start from |
| Isaac Lab | commit `37ddf62` (2.3.2) | BSD-3-Clause | simulation, robot configuration, differential IK |
| Isaac Sim | 5.1 | NVIDIA license | physics and rendering |
| `unitreerobotics/unitree_ros` | commit `7d6075f` | BSD-3-Clause | the official G1 description with Brainco Revo2 hands |
| Isaac Sim G1 asset (`g1.usd`) | shipped with 5.1 | NVIDIA license | the G1 with Dex3 hands used in Stage A |
| LeRobot v2 dataset layout | format only | Apache-2.0 | what the GR00T data loader expects |

Vendor pages for the Revo2 (mass, grip force, closing time, bus protocols and the Touch sensing)
are cited in [hand_models.md](hand_models.md) and marked unverified, since they were not checked
against primary documentation.
