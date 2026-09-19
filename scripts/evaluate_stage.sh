#!/usr/bin/env bash
# Closed-loop evaluation of one stage over the five test configurations, 10 trials each.
# One simulator process per configuration (the object geometry is fixed when the scene is built).
#
# usage: bash scripts/evaluate_stage.sh <A|B|C> <dex3|brainco> <none|dex3_to_brainco> <port> <trials> [configurations...]
#   Stage A: bash scripts/evaluate_stage.sh A dex3    none            5601 10
#   Stage B: bash scripts/evaluate_stage.sh B brainco dex3_to_brainco 5601 10
#   Stage C: EXTRA="--max-steps 1050" bash scripts/evaluate_stage.sh C brainco none 5612 10
#
# Requires a policy server (slurm/policy_server.sbatch) reachable on localhost:<port>.
# EXTRA: additional arguments for handover_rollout.py. The episode cap is about 1.1x the length
# of the demonstrations the evaluated policy was trained on (Dex3 680 steps -> 750 by default,
# Brainco 940 steps -> 1050).
# env: ISAAC_PY (python of the Isaac Lab install, default: python)
set -u
STAGE=$1; ROBOT=$2; RETARGET=$3; PORT=$4; TRIALS=$5; shift 5
CONFIGS=${*:-"stick_30x240 stick_30x220 stick_35x240 heldout_object_round heldout_initial_pose"}
ISAAC_PY=${ISAAC_PY:-python}
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
mkdir -p logs "results/stage$STAGE"
COMMON="--robot $ROBOT --retarget $RETARGET --stage-label $STAGE --port $PORT --episodes $TRIALS \
  --object cylinder --stick 1 --place-xy -0.14 0.46 --pose-noise 0.02 --headless ${EXTRA:-}"

stop_sim() {
  if command -v taskkill >/dev/null 2>&1; then taskkill //IM python.exe //F >/dev/null 2>&1
  else pkill -f "scripts/handover_rollout.py" >/dev/null 2>&1; fi
  return 0
}

for cfg in $CONFIGS; do
  case $cfg in
    stick_30x240)         X="--stick-w 0.030 --stick-h 0.24 --obj-start 0.10 0.32 --seed 7100" ;;
    stick_30x220)         X="--stick-w 0.030 --stick-h 0.22 --obj-start 0.10 0.32 --seed 7200" ;;
    stick_35x240)         X="--stick-w 0.035 --stick-h 0.24 --obj-start 0.10 0.32 --seed 7300" ;;
    heldout_object_round) X="--stick-w 0.030 --stick-h 0.24 --stick-round 1 --obj-start 0.10 0.32 --seed 7400" ;;
    heldout_initial_pose) X="--stick-w 0.030 --stick-h 0.24 --obj-start 0.13 0.29 --pose-noise 0.0 --seed 7500" ;;
    *) echo "unknown configuration $cfg"; continue ;;
  esac
  OUT="results/stage$STAGE/$cfg"
  [ -f "$OUT/results_$STAGE.json" ] && { echo "skip $cfg (already evaluated)"; continue; }
  echo "Stage $STAGE, $cfg -> $OUT"
  for attempt in 1 2 3 4 5; do     # exit code 3 = cameras came up black, relaunch
    rm -rf "$OUT"
    timeout 5400 $ISAAC_PY scripts/handover_rollout.py $COMMON $X --out "$OUT" > "logs/eval_${STAGE}_$cfg.log" 2>&1
    rc=$?
    stop_sim
    [ $rc -eq 3 ] || break
    echo "  cameras came up black, relaunching ($attempt)"; sleep 5
  done
  grep -E "HROLL_SUCCESS_RATE|Traceback" "logs/eval_${STAGE}_$cfg.log" | tail -1
done
echo "Stage $STAGE evaluation done"
