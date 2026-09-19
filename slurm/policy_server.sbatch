#!/bin/bash
#SBATCH --job-name=gr00t_policy_server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=logs/policy_server_%j.out
#
# Serve one fine-tuned checkpoint to the simulator (scripts/handover_rollout.py) over TCP.
#
#   sbatch -A <account> -p <gpu_partition> \
#     --export=ALL,CKPT=<checkpoint dir>,MODCFG=scripts/dex3_handover_config.py,PORT=5601 slurm/policy_server.sbatch
#
# MODCFG is scripts/dex3_handover_config.py for the Stage A checkpoint (also used by Stage B,
# with the retargeting layer on the simulator side) and scripts/brainco_config.py for Stage C.
# When the log prints POLICY_SERVER_READY, forward the port to the simulation machine:
#   ssh -N -L <port>:<gpu node>:<port> <cluster login host>
set -euo pipefail
: "${CKPT:?set CKPT to a checkpoint directory}"
: "${MODCFG:?set MODCFG to the modality config used in training}"
PORT=${PORT:-5601}
SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
ROOT=${PROJECT_ROOT:-$SUBMIT}
REPO=${GR00T_REPO:-$ROOT/Isaac-GR00T}
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
echo "POLICY_SERVER_NODE $(hostname -s)"
cd "$REPO"
python "$SUBMIT/scripts/policy_server.py" \
  --model-path "$CKPT" \
  --modality-config-path "$SUBMIT/$MODCFG" \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port "$PORT"
