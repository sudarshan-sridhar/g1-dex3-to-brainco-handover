# Assignment 2: low-data fine-tuning and sim-to-real transfer for a new capability

A written pipeline, as the assignment asks. No code. Numbers marked "to measure" are assumptions.
Everything else comes from the Assignment 1 pipeline in this repository.

## 1. The new task

Standing bimanual bin-to-bin sorting with tactile-confirmed grasps. Two bins sit on a table with a
mixed bin of six to eight small household objects (cups, cans, blocks, a soft ball, a bottle). The
G1 stands unsupported, picks each object with the nearer hand, sorts it by a spoken instruction
("rigid objects to the left bin, soft ones to the right"), and drops it in the matching bin.
Objects near the midline are passed from one hand to the other, which reuses the handover skill
from Assignment 1.

The task extends Assignment 1 along the axes it left open. It needs a standing controller, where
Assignment 1 allowed a fixed base. It needs object variety and language conditioning, which is
evidence the behaviour is not tied to one trajectory. And it is the first task where the Touch
fingertip sensing carries weight: a pick counts only when the pads confirm contact, and softness is
judged from the pressure profile during closing. Drawer opening was considered and rejected as
single-handed with no use for tactile sensing; tray carrying was rejected because it exercises no
grasp variety.

Success is the fraction of objects placed in the correct bin, alongside the same metrics as
Assignment 1: drops, falls, unintended contacts, timing, adaptation cost and failure categories.

## 2. System architecture

```
head camera + chest camera + joint state + 10 fingertip pressures + IMU
        |
        v
GR00T N1.7-3B (vision-language backbone frozen; projector and action head trained)
   state:  14 arm + 12 hand + 10 tactile + 3 torso
   action: 16 steps of (14 arm + 12 hand + 6 upper-body reference), 20 Hz
        |                                  |
        v                                  v
 executor at 500 Hz                  SONIC whole-body controller
 (interpolation, tactile grasp gate)  (balance, legs and waist, 500 Hz)
        |                                  |
        +----------> robot low-level command bus <----------+
```

GR00T owns the 14 arm joints and 12 hand joints. SONIC owns the 12 leg joints and 3 waist joints.
The interface between them is a six-value upper-body reference (torso height and pitch, and a
reach command per wrist in the pelvis frame) that GR00T emits as extra action dimensions and SONIC
consumes as its motion command, in place of the motion token its own G1 configuration uses. Arm
joint positions are also fed to SONIC as observations so balance anticipates the arm mass shift.

## 3. Adapting the pretrained stack

- **GR00T.** Start from the Assignment 1 Stage C checkpoint, which already holds the handover
  skill on Brainco hands. Extend the state with ten normalised pad pressures and the torso pose,
  and the action with the six-value SONIC reference. Keep the same freeze split: backbone and
  vision encoder frozen, projector and action head trained. The slot widens from 26 to 32 action
  values; GR00T pads actions to 132 internally, so the checkpoint loads and only the new columns
  start untrained.
- **SONIC.** Use the released G1 motion-tracking policy as is for balance. Fine-tune it only if
  the arm-driven torso reference falls outside its training distribution, judged by tracking error
  on simulated reference trajectories (thresholds of 5 cm and 5 degrees, to measure). If needed,
  train legs and waist only on the standing-reach subset with the upper-body reference sampled
  from the GR00T demonstration distribution.
- **Tactile gate.** Outside both networks, in the executor: closing advances to lifting only when
  at least two pads on the grasping hand exceed about 1 N for three consecutive control steps, and
  a loss of all pad pressure during the carry halts the arm and logs a drop. This is the safety
  gate from the sim-to-real plan, promoted to task logic.

## 4. Simulation fine-tuning

Three stages, each starting from the previous checkpoint: fixed base without tactile, to check the
sorting behaviour; fixed base with simulated tactile; then standing with the full stack. Tactile in
simulation comes from contact sensors on the ten fingertip links, with net normal force mapped to
pressure. Shear and proximity are not simulated and are zeroed with state dropout on that group so
the policy does not learn to depend on them.

Budget per stage, scaled from Assignment 1, where 10,000 steps at batch 16 took 2 h 12 min on one
A40: about 10,000 steps per stage, checkpoint every 2,000, select by closed-loop success on three
configurations rather than by loss. Hold out one object (the bottle) and one bin layout (bins
swapped) and report them separately.

## 5. Synthetic data

- **Scripted generator first**, extending `scripts/handover_demos.py`: a grasp offset table per
  object and a phase script (reach, close under the tactile gate, lift, carry, drop). Target about
  300 successful episodes per stage (to measure). The Assignment 1 generator yields 68 to 98
  percent depending on hand and object, at about 3 minutes per episode.
- **Multiply with Isaac Lab Mimic**: annotate the scripted episodes into subtasks and re-synthesise
  them at new object poses for five to ten times the seed set, keeping successes only.
- **Randomise** object pose (5 cm and any yaw), scale 0.9 to 1.1, mass 30 to 300 g, friction 0.4 to
  1.0, table height 2 cm, camera pose 2 cm and 2 degrees, lighting, texture, and tactile gain and
  noise.
- **Objects**: eight training objects from simple primitives plus the held-out bottle. Soft objects
  are rigid meshes with a lower tactile threshold, since deformables are not in this pipeline.

Two lessons from Assignment 1 apply directly to any generated dataset: drop episodes whose cameras
failed to render, and pair each observation with the next command rather than the one already
applied.

## 6. Minimum real and teleoperated data

Collect with the Unitree teleoperation stack driving the Brainco hands, logging the same state and
action layout at 20 Hz plus pad pressures and both cameras. Sixty episodes, roughly twenty per
object class, is about two hours of operator time including resets (to measure). The reasoning is
that the simulated policy already carries the motor skill; real data is needed to align the visual
and tactile projector to real cameras and real pads. Hold out ten episodes for selection.

Then co-fine-tune on simulated and real data with the real episodes upweighted about three times,
backbone still frozen. If bench success stays below half after that, collect another sixty episodes
before changing anything else.

## 7. Sim-to-real

Reuse sections 1 to 5 of the [sim-to-real plan](sim_to_real_plan.md): camera and hand calibration,
20 Hz policy over a 500 Hz bridge, latency under 400 ms with on-robot inference, joint and
workspace limits, and staged validation on the stand. Additions for this task:

- Calibrate each pad against known 1 N, 5 N and 10 N loads and verify the gate thresholds on the
  bench before any arm motion.
- Enable the standing controller only after the full task passes on the stand at 7 of 10 (to
  measure). The first standing trials use a gantry tether and a workspace box shrunk by 10 cm.
- Order of validation: hands on the bench, replay on the stand, closed loop on the stand, closed
  loop standing with a tether, then untethered. Report simulated and real metrics side by side and
  list the gap per metric.
