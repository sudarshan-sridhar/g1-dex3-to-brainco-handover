# Robot assets

The robot description and the converted simulation asset are not stored in this repository. Both
are reproduced with two commands.

```bash
git clone https://github.com/unitreerobotics/unitree_ros assets_brainco/unitree_ros
git -C assets_brainco/unitree_ros checkout 7d6075f
python scripts/convert_brainco_urdf.py     # writes assets_brainco/usd/g1_29dof_brainco.usd
```

`unitree_ros` is BSD-3-Clause and carries
`robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf`, the official G1_29DoF
description with Brainco Revo2 hands. The converter applies the fixes described in
[../docs/hand_models.md](../docs/hand_models.md).

The Dex3 version of the robot is the `g1.usd` asset shipped with Isaac Sim 5.1 and is loaded from
the Isaac Sim asset root, so nothing has to be downloaded for Stage A.

A converted copy of the Brainco asset is attached to the release for convenience.
