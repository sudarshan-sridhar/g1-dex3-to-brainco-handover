#!/usr/bin/env bash
# Merge demonstration folders into one LeRobot v2 dataset for GR00T fine-tuning.
#
# Only successful demonstrations are kept. Episodes whose cameras rendered black are dropped.
# Each observation is paired with the next joint command (--action-shift 1), because the
# recorder stores the robot state and camera frames after applying the command of that step.
#
# usage: bash scripts/build_dataset.sh <A|C> <demo_dir> <dataset_name>
#   A: Dex3 hands, 28-D state/action     C: Brainco hands, 26-D state/action
# env: ISAAC_PY (python with numpy, pandas, pyarrow, imageio-ffmpeg; default: python)
set -euo pipefail
STAGE=$1; DEMOS=$2; NAME=$3
cd "$(dirname "$0")/.."
ISAAC_PY=${ISAAC_PY:-python}
TASK="hand the object from the right hand to the left hand and place it on the left"
if [ "$STAGE" = "A" ]; then DIM=28; ROBOT=unitree_g1_dex3_handover; else DIM=26; ROBOT=unitree_g1_brainco_handover; fi

MERGED="data/${NAME}_merged"
rm -rf "$MERGED" "data/$NAME"
$ISAAC_PY scripts/merge_demo_dirs.py --drop-black --out "$MERGED" $(ls -d "$DEMOS"/*/ | sed 's:/$::')
$ISAAC_PY scripts/convert_handover_to_lerobot.py --dim $DIM --robot-type $ROBOT --task-text "$TASK" \
  --fps 20 --action-shift 1 --in "$MERGED" --out "data/$NAME" --keep success
echo "dataset ready: data/$NAME"
echo "copy it to the training machine, for example: rsync -a data/$NAME <host>:<project>/datasets/"
