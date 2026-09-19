# Sim-to-real deployment plan

Moving the Stage C policy (GR00T N1.7-3B, Brainco slot, 26 joint targets at 20 Hz, 16 steps per
inference) from Isaac Lab onto a physical G1_29DoF with Brainco Revo2 Touch hands. None of this
has been run on hardware. Values marked "to measure" are assumptions that must be checked on the
real system; the rest come from the simulation pipeline or from vendor documentation.

## 1. Calibration

**Cameras.** In simulation the head camera sits at (0, 0.10, 1.50) in the robot frame looking at
(0, 0.30, 1.06), and the front camera at (0.75, 1.15, 1.60) looking at (0, 0.28, 1.10), both
224x224. On hardware: calibrate the head camera intrinsics with a checkerboard (target
reprojection error below 0.5 px, to measure), mount a second camera at the front pose, calibrate
both against the pelvis frame with a marker on the torso, and crop to the simulated field of view.
The policy has only ever seen the simulated placement, so a mismatch beyond about 5 cm or 5 degrees
should be corrected by moving the camera rather than in software.

**Hands.** The Revo2 reports normalised motor positions over its bus while the policy speaks
radians with limits 0 to 1.5184 (thumb swing), 0 to 1.0472 (thumb bend) and 0 to 1.4661 (fingers).
Drive each motor to both ends, record the bus values, and fit a linear map per joint. Confirm on
the real hand that increasing command closes the finger on both hands, which is what the official
description implies.

**Arms.** Compare the encoder zero pose against the simulated default pose and store a per-joint
offset.

**Workspace.** In simulation the table top sits about 6 mm below the pinned pelvis height, the
stick starts at (0.10, 0.32) in the robot frame and the target is 28 cm away on the left. Measure
the real table height from the pelvis, shim it to within 2 cm (to measure), and mark the start and
target positions.

## 2. Control rate and chunk streaming

The policy produces one 26-value target every 50 ms. The G1 low-level interface takes commands at
500 Hz, so a bridging node holds the newest target and interpolates between consecutive targets,
which matches the simulated behaviour of holding each target for six physics steps. Hand targets
go to the Revo2 bus at its own rate, assumed 100 Hz (to measure), interpolated the same way.

Each inference returns 16 targets, which is 0.8 s of motion. The executor plays them from a
buffer and requests the next chunk when 8 remain, so 0.4 s of inference latency is hidden. If the
next chunk is late, the last target is held rather than extrapolated. No temporal ensembling: it
reduced success in earlier simulated runs.

## 3. Latency budget

Target: under 400 ms per inference, so the 0.4 s lookahead covers it.

| path | estimate |
|:--|:--|
| camera capture and resize | about 30 ms at 30 fps |
| joint state read | under 2 ms |
| inference, GR00T N1.7-3B in fp16 on an on-robot Jetson | 60 to 100 ms (to measure) |
| inference, workstation GPU over wired Ethernet | about 40 ms plus under 5 ms of network |
| hand bus write | about 10 ms (to measure) |

The setup used for the simulated results, a policy on a shared cluster reached over SSH, measured
150 to 250 ms per chunk with jitter that depends on the campus network. That is fine for
development and not acceptable for hardware.

Log the timestamp of every observation and applied target, and report the achieved rate and the
fraction of late chunks alongside the simulated numbers.

## 4. Safety limits

- Arm joint velocity clamped to 1.0 rad/s and torque to half the rated effort in the bridging
  node, with any target beyond a limit plus 2 degrees clipped and counted.
- A workspace box on both wrists in the pelvis frame, taken from the demonstration waypoints plus
  10 cm: x in [-0.45, 0.45], y in [0.05, 0.55], z in [-0.15, 0.45] (to measure). Leaving the box
  freezes the arm.
- Hand force: run the Revo2 in current mode with a cap equivalent to about 10 N at the fingertip,
  raised to 20 N only while an object is held. Position mode for opening motions.
- Hardware emergency stop plus a software kill that zeroes torque and opens both hands. If no
  fresh chunk arrives within 1 s the executor holds; after 3 s it retreats to a neutral pose at
  0.3 rad/s.
- The robot stays on its stand with the legs in damping mode, which matches the fixed base used in
  simulation. A standing controller is out of scope until the stages below pass on the stand.

## 5. Staged hardware validation

1. **Hands on the bench.** Replay the 12 hand columns of recorded demonstrations through the
   calibration map. Pass: no over-current, closure times near the rated 0.65 s, motion matches the
   simulated video.
2. **Arms and hands, no object.** Robot on the stand, empty table, open-loop replay of full
   demonstrations. Pass: wrist tracking within 3 cm RMS of the simulated trajectory (to measure),
   no workspace or limit trips.
3. **Single-hand grasp.** Right hand only, the same stick, open-loop replay. Pass: lifted and held
   for 5 s in 8 of 10 attempts (to measure). Tune the current cap here.
4. **Handover, open loop.** Full demonstration replay with a soft stick. Pass: transfer without a
   drop in 7 of 10 (to measure), with drops and unintended contacts logged as in simulation.
5. **Closed loop.** Stage C checkpoint on the on-robot GPU, operator on the emergency stop,
   workspace box and force caps active, first ten trials at half speed. Run the same five
   configurations as in simulation and report the same metrics side by side with the simulated
   numbers, listing the gap per metric.

## 6. Using the Touch sensing

The fingertip pressure, shear and proximity of the Touch variant are not in any available
simulation asset, so the policy never sees them. Use them first outside the policy, as a grasp
gate in the executor: advance from closing to lifting only when at least two fingertips report
pressure above about 1 N (to measure); allow the right hand to open only once the left hand's pads
report contact; treat a sudden loss of pressure during the carry as a drop, hold, and log it.

Later, add the ten pad pressures as an extra state group and fine-tune the projector on real data
with the backbone still frozen. Simulated data then carries zeros for that group with state
dropout, so the slot does not learn to depend on a signal that simulation cannot provide.
