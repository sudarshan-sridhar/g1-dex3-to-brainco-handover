# Cartesian tracking error (M2)

Distance between the commanded pose and the pose reached one control step later, from
forward kinematics on the commanded and the measured joint values. Averaged over every
fifth control step of every trial.

| stage | configuration | wrist tracking error, mean / max (mm) | fingertip tracking error, mean / max (mm) |
|---|---|---|---|
| A | stick_30x240 | 4.9 / 37.6 | n/a (Dex3 hand geometry is USD only) |
| A | stick_30x220 | 4.7 / 26.4 | n/a (Dex3 hand geometry is USD only) |
| A | stick_35x240 | 5.0 / 40.2 | n/a (Dex3 hand geometry is USD only) |
| A | heldout_object_round | 5.3 / 64.4 | n/a (Dex3 hand geometry is USD only) |
| A | heldout_initial_pose | 5.0 / 79.5 | n/a (Dex3 hand geometry is USD only) |
| B | stick_30x240 | 5.4 / 152.1 | 0.6 / 28.4 |
| B | stick_30x220 | 4.2 / 72.9 | 0.5 / 46.2 |
| B | stick_35x240 | 4.5 / 108.8 | 0.4 / 30.2 |
| B | heldout_object_round | 4.5 / 102.2 | 0.5 / 30.6 |
| B | heldout_initial_pose | 4.4 / 37.6 | 0.4 / 28.6 |
| C | stick_30x240 | 3.9 / 48.7 | 1.2 / 36.8 |
| C | stick_30x220 | 4.1 / 31.3 | 1.8 / 39.6 |
| C | stick_35x240 | 3.6 / 43.6 | 1.5 / 40.7 |
| C | heldout_object_round | 4.2 / 43.9 | 1.1 / 51.4 |
| C | heldout_initial_pose | 3.9 / 36.6 | 0.8 / 46.9 |
