#!/bin/bash
#SBATCH --job-name=gr00t_stage_a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/finetune_stage_a_%j.out
#
# Stage A: fine-tune GR00T N1.7-3B on the scripted Dex3 handover demonstrations.
# Reported run: one NVIDIA A40 (48 GB), 16,000 steps at batch 16, 3 h 29 min of training.
#
# Submit from the repository root with your cluster's account and GPU partition:
#   sbatch -A <account> -p <gpu_partition> slurm/finetune_stage_a.sbatch
#
# Environment (override with --export=ALL,NAME=value):
#   PROJECT_ROOT  folder holding datasets/ and checkpoints/          (default: submit directory)
#   GR00T_REPO    clone of NVIDIA/Isaac-GR00T at commit 51d4c89     (default: $PROJECT_ROOT/Isaac-GR00T)
#   VENV          python environment with GR00T installed, see env/gr00t_requirements.txt
#   CUDA_HOME     CUDA 12.8 toolkit; DeepSpeed needs nvcc when it is imported
#   DATA_DIR, OUT_DIR, MAX_STEPS
set -euo pipefail
SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
ROOT=${PROJECT_ROOT:-$SUBMIT}
REPO=${GR00T_REPO:-$ROOT/Isaac-GR00T}
DATA=${DATA_DIR:-$ROOT/datasets/lerobot_dex3_handover}
OUT=${OUT_DIR:-$ROOT/checkpoints/gr00t_n17_g1_dex3_handover}
MODCFG=$SUBMIT/scripts/dex3_handover_config.py
MAX_STEPS=${MAX_STEPS:-16000}

[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
[ -n "${CUDA_HOME:-}" ] && export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<PY
import json
info = json.load(open("$DATA/meta/info.json")); mod = json.load(open("$DATA/meta/modality.json"))
print("dataset:", info["total_episodes"], "episodes,", info["total_frames"], "frames, fps", info["fps"])
assert info["fps"] == 20, "the handover data is recorded at 20 Hz"
assert mod["action"]["hands"]["end"] == 28, "Stage A expects the 28-D Dex3 layout"
PY

cd "$REPO"
# statistics for the new embodiment are computed from this dataset
[ -f "$DATA/meta/relative_stats.json" ] || python gr00t/data/stats.py \
  --dataset-path "$DATA" --embodiment-tag NEW_EMBODIMENT --modality-config-path "$MODCFG"

RESUME=""
LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[ -n "$LAST" ] && RESUME="--resume-from-checkpoint $LAST" && echo "resuming from $LAST"

# Updated: backbone-to-head projector and the flow-matching action head (DiT), including the
# NEW_EMBODIMENT state/action encoders. Frozen: Eagle language model and vision encoder.
torchrun --nproc_per_node=1 --master_port=29521 gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
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
  --save-steps 4000 \
  --save-total-limit 12 \
  --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  $RESUME
ls -d "$OUT"/checkpoint-*
