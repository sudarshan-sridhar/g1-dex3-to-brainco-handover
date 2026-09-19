#!/usr/bin/env bash
# Scripted handover demonstrations for the three training configurations.
#
# Each simulator process records 5 episodes. If Isaac Sim starts with cameras that render
# black, handover_demos.py exits with code 3 and that chunk is relaunched.
#
# usage:
#   DEMO_ARGS="$(cat configs/demo_args_dex3.txt)"    STAGE=A bash scripts/generate_demos.sh data/demos_dex3    5
#   DEMO_ARGS="$(cat configs/demo_args_brainco.txt)" STAGE=C bash scripts/generate_demos.sh data/demos_brainco 3
#
# env: ISAAC_PY (python of the Isaac Lab install, default: python), SEED_OFFSET (default 0)
set -u
OUT=$1; NCHUNKS=$2; shift 2
CONFIGS=${*:-"stick_30x240 stick_30x220 stick_35x240"}
ISAAC_PY=${ISAAC_PY:-python}
STAGE=${STAGE:-A}
SEED_OFFSET=${SEED_OFFSET:-0}
: "${DEMO_ARGS:?set DEMO_ARGS to the contents of configs/demo_args_dex3.txt or configs/demo_args_brainco.txt}"
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
mkdir -p "$OUT" logs

stop_sim() {  # make sure no simulator process is left behind between chunks
  if command -v taskkill >/dev/null 2>&1; then taskkill //IM python.exe //F >/dev/null 2>&1
  else pkill -f "scripts/handover_demos.py" >/dev/null 2>&1; fi
  return 0
}

for cfg in $CONFIGS; do
  case $cfg in
    stick_30x240) X="--stick-w 0.030 --stick-h 0.24 --obj-start 0.10 0.32"; SEED0=8100 ;;
    stick_30x220) X="--stick-w 0.030 --stick-h 0.22 --obj-start 0.10 0.32"; SEED0=8200 ;;
    stick_35x240) X="--stick-w 0.035 --stick-h 0.24 --obj-start 0.10 0.32"; SEED0=8300 ;;
    *) echo "unknown configuration $cfg"; continue ;;
  esac
  for k in $(seq 0 $((NCHUNKS - 1))); do
    D="$OUT/${cfg}_chunk$(printf %02d "$k")"
    LOG="logs/demos_${STAGE}_${cfg}_chunk$k.log"
    [ -f "$D/handover_summary.json" ] && continue
    for attempt in 1 2 3 4; do
      rm -rf "$D"
      timeout 2400 $ISAAC_PY scripts/handover_demos.py $DEMO_ARGS $X --stage-label "$STAGE" \
        --episodes 5 --seed $((SEED0 + SEED_OFFSET + 10 * k)) --out "$D" > "$LOG" 2>&1
      rc=$?
      stop_sim
      [ $rc -eq 3 ] || break
      echo "  $cfg chunk $k: cameras came up black, relaunching ($attempt)"
      sleep 5
    done
    grep -h "HANDOVER_SUCCESS_RATE" "$LOG" | sed "s/^/$cfg chunk $k: /"
  done
done
echo "done: $OUT"
