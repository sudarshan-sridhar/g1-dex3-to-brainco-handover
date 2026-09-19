#!/bin/bash
#SBATCH --job-name=gr00t_stage_c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/finetune_stage_c_%j.out
#
# Stage C: fine-tune the Stage A checkpoint on the scripted Brainco handover demonstrations.
# Reported run: one NVIDIA A40 (48 GB), 10,000 steps at batch 16, 2 h 12 min of training.
#
# The starting point is the Stage A checkpoint, not the base model, so the Dex3 handover
# behaviour of the arms is kept and the hand behaviour is learned for the new hand. The action
# layout changes from 28-D to 26-D (GR00T pads actions to 132 dims, so the checkpoint loads);
# normalisation statistics are recomputed for the 26-D dataset.
#
# Submit from the repository root:
#   sbatch -A <account> -p <gpu_partition> slurm/finetune_stage_c.sbatch
#
# Environment (override with --export=ALL,NAME=value):
#   PROJECT_ROOT, GR00T_REPO, VENV, CUDA_HOME   as in finetune_stage_a.sbatch
#   BASE_CKPT   Stage A checkpoint   (default: $PROJECT_ROOT/checkpoints/gr00t_n17_g1_dex3_handover/checkpoint-16000)
#   DATA_DIR, OUT_DIR, MAX_STEPS
set -euo pipefail
SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
ROOT=${PROJECT_ROOT:-$SUBMIT}
REPO=${GR00T_REPO:-$ROOT/Isaac-GR00T}
BASE=${BASE_CKPT:-$ROOT/checkpoints/gr00t_n17_g1_dex3_handover/checkpoint-16000}
DATA=${DATA_DIR:-$ROOT/datasets/lerobot_brainco_handover}
OUT=${OUT_DIR:-$ROOT/checkpoints/gr00t_n17_g1_brainco_handover}
MODCFG=$SUBMIT/scripts/brainco_config.py
MAX_STEPS=${MAX_STEPS:-10000}

[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
[ -n "${CUDA_HOME:-}" ] && export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
test -d "$BASE" || { echo "Stage A checkpoint not found: $BASE"; exit 1; }
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<PY
import json
info = json.load(open("$DATA/meta/info.json")); mod = json.load(open("$DATA/meta/modality.json"))
print("dataset:", info["total_episodes"], "episodes,", info["total_frames"], "frames, fps", info["fps"])
assert info["fps"] == 20, "the handover data is recorded at 20 Hz"
assert mod["action"]["hands"]["end"] == 26, "Stage C expects the 26-D Brainco layout"
PY

cd "$REPO"
[ -f "$DATA/meta/relative_stats.json" ] || python gr00t/data/stats.py \
  --dataset-path "$DATA" --embodiment-tag NEW_EMBODIMENT --modality-config-path "$MODCFG"

RESUME=""
LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[ -n "$LAST" ] && RESUME="--resume-from-checkpoint $LAST" && echo "resuming from $LAST"

# Updated: projector and flow-matching action head (DiT) with the NEW_EMBODIMENT encoders.
# Frozen: Eagle language model and vision encoder. Same split as Stage A.
torchrun --nproc_per_node=1 --master_port=29531 gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path "$DATA" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODCFG" \
  --output-dir "$OUT" \
  --num-gpus 1 \
  --max-steps "$MAX_STEPS" \
  --global-batch-size 16 \
  --learning-rate 1e-4 \
  --state-dropout-prob 0.05 \
  --dataloader-num-workers 4 \
  --save-steps 2000 \
  --save-total-limit 12 \
  --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  $RESUME
ls -d "$OUT"/checkpoint-*
