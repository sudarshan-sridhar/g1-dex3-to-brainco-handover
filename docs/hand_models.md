# Hand models and the asset swap

Covers assignment items 1.1 (identify both hands) and 1.2 (replace the hand model in the
simulation asset). Vendor figures that were not checked against a primary document are marked
"unverified".

## 1.1 Source and target hands

| | source: Unitree Dex3-1 | target: Brainco Revo2 Touch |
|:--|:--|:--|
| mounted on | Unitree G1_29DoF (`g1.usd` shipped with Isaac Sim 5.1) | Unitree G1_29DoF (`g1_29dof_mode_15_brainco_hand.urdf` in `unitree_ros`) |
| fingers | 3 (thumb, index, middle) | 5 |
| joints per hand | 7, all actuated, one motor each | 16 revolute in the file: 6 actuated, 5 distal driven by linkage, 5 degenerate tip joints |
| thumb | yaw plus two bend joints | swing across the palm (0 to 1.5184 rad) plus one bend (0 to 1.0472); the distal joint follows at ratio 1.0 |
| finger | two independent joints | one bend (0 to 1.4661); the distal joint follows at ratio 1.155 |
| joint limits | mirrored between hands by sign; soft limits read from the asset at run time | all 0 to upper on both hands, mirrored by joint origin, so positive closes on both sides |
| actuation | direct drive, one motor per joint | 6 motors per hand; URDF efforts 0.5, 1.1 and 2.0 Nm, velocities 2.27 to 2.62 rad/s |
| sensing | joint encoders; fingertip pressure on the sensor variant (unverified) | joint position; the Touch variant adds fingertip pressure, shear and proximity (unverified, and not modelled in any available asset) |
| control mode on hardware | joint position targets over the Unitree bus (unverified) | position, position with speed, speed, current and PWM over RS485, CAN-FD or EtherCAT (unverified) |
| control in simulation | absolute joint position targets, implicit PD | same, with the distal joints written from their proximal joint every step |
| mass, grip force | not documented locally | 383 g per hand, 50 N fist grip, 0.65 s to close (vendor figures, unverified) |
| used here | 14 values per robot (7 per hand) inside a 28-value action | 12 values per robot (6 per hand) inside a 26-value action |

## 1.2 Replacing the hand in the asset

Source file: `unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf`,
BSD-3-Clause, commit `7d6075f`. `scripts/convert_brainco_urdf.py` converts it to USD with the
Isaac Lab importer. Everything below the two `wrist_yaw_link` frames comes from that file: link
frames, masses, inertias, collision meshes (each link reuses its visual mesh), joint axes, limits
and the distal transmissions. The 29 body joints above the wrists are identical to the Dex3 model,
so the arms, torso and legs are unchanged and the arm controller needs no adjustment.

Three things in the file need fixing during conversion, all handled by the converter:

1. **Degenerate tip joints.** All ten `*_tip*` joints are declared revolute with equal lower and
   upper limits. Imported as they are, the physics engine snaps four of them to 1 rad on the first
   step and the fingertips end up bent sideways. They are rewritten as fixed joints.
2. **Naming.** Link and joint names differ in case between the two hands, and the right thumb tip
   drops the `_joint` suffix the other nine carry. Body-name patterns have to match both spellings.
3. **Distal transmissions.** The importer is set to convert mimic joints into ordinary joints, and
   `scripts/brainco_cfg.py` then enforces `distal = ratio x proximal` (1.0 for the thumb, 1.155 for
   the fingers) every time a hand target is written, clipped to the distal limit. The alternative,
   leaving them as physics mimic joints, leaves them undriven and they fight stray targets.

Actuator settings in simulation (`scripts/brainco_cfg.py`): implicit PD, stiffness 10, damping 0.5,
effort limit twice the URDF motor rating, velocity limit 5 rad/s. Damping is deliberately slightly
below critical for these very light links so the distal phalanges do not lag their target. The Dex3
hands keep the gains Isaac Lab ships, which are considerably stiffer; that asymmetry is listed as a
limitation in the [technical report](technical_report.md).

Scene geometry unchanged between hands: same table, same object set, same camera poses, same
target marker, same initial arm pose. Only the hands differ, which is what makes the Stage A to
Stage B comparison meaningful.
