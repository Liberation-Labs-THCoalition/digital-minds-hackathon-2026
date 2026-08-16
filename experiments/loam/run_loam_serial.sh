#!/bin/bash
# Loam serial runner — one process per arm, fresh MPS each time.
# Avoids the MPS crash that kills long-running sessions.
#
# Usage on Starship:
#   JLENS_PATH=~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
#       bash experiments/loam/run_loam_serial.sh 20 data/loam_serial

QUADS=${1:-20}
DATA_DIR=${2:-data/loam_serial}
PYTHON=${PYTHON:-~/miniforge/envs/oracle/bin/python}
SCRIPT="experiments/loam/run_loam.py"
SEED=42

mkdir -p "$DATA_DIR"
echo "=== LOAM SERIAL: $QUADS quads, output $DATA_DIR ==="
echo "Started: $(date)"

COMPLETED=0
FAILED=0

for q in $(seq 1 "$QUADS"); do
  for arm in enacted observed briefed null; do
    echo ""
    echo "--- Quad $q / $arm ($(date)) ---"

    # Skip if already done
    if [ -f "$DATA_DIR/quad_$(printf '%02d' $q)/$arm/event_log.json" ]; then
      echo "  Already complete, skipping."
      continue
    fi

    # Skip observed if no enacted event log
    if [ "$arm" = "observed" ] && [ ! -f "$DATA_DIR/quad_$(printf '%02d' $q)/enacted/event_log.json" ]; then
      echo "  No enacted event log, skipping observed."
      continue
    fi

    # Fresh process per arm
    $PYTHON "$SCRIPT" \
      --arm "$arm" \
      --quad "$q" \
      --seed "$SEED" \
      --data-dir "$DATA_DIR" \
      2>&1 | tail -5

    if [ $? -eq 0 ]; then
      echo "  DONE: quad $q $arm"
      COMPLETED=$((COMPLETED + 1))
    else
      echo "  FAILED: quad $q $arm"
      FAILED=$((FAILED + 1))
    fi

    # Brief pause for MPS cleanup
    sleep 5
  done

  echo ""
  echo "=== Quad $q complete. Completed: $COMPLETED, Failed: $FAILED ==="
done

echo ""
echo "=== LOAM SERIAL COMPLETE ==="
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Finished: $(date)"
