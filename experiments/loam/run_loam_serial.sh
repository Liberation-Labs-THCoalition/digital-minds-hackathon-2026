#!/bin/bash
# Loam serial runner, Run 3 (Amendment 2). One process per session, fresh MPS each time.
#
# Usage on Starship, from the repo root:
#   JLENS_PATH=~/jlens-community/lenses/qwen3.5-27b_jlens.pt \
#       bash experiments/loam/run_loam_serial.sh 20 data/loam_run3
#
# A session is COMPLETE when its event log has the end of the recall phase followed by aftercare
# (experiments/loam/session_state.py). A session whose agent declines or withdraws is FINAL and is never rerun.
# Anything else is a TECHNICAL stop: the attempt is moved aside (kept, for the report) and the session is rerun from
# the start in a fresh process, up to MAX_RETRIES times, with the same seed. A session that stops making progress
# for STALL_S seconds is killed and counts as a technical stop. The observed arm runs only once its enacted session
# is COMPLETE or FINAL.
#
# Every attempt is a row in $DATA_DIR/sessions.tsv; each attempt's full output is in $DATA_DIR/logs/.

set -u
QUADS=${1:-20}
DATA_DIR=${2:-data/loam_run3}
FIRST_QUAD=${FIRST_QUAD:-1}
PYTHON=${PYTHON:-$HOME/miniforge/envs/oracle/bin/python}
SCRIPT=${SCRIPT:-experiments/loam/run_loam.py}
STATE=${STATE:-experiments/loam/session_state.py}
SEED=42
MAX_RETRIES=${MAX_RETRIES:-2}
STALL_S=${STALL_S:-1800}
POLL_S=${POLL_S:-30}
CLEANUP_S=${CLEANUP_S:-5}
ARMS=${ARMS:-"enacted observed briefed null"}

mkdir -p "$DATA_DIR/logs"
TSV="$DATA_DIR/sessions.tsv"
[ -f "$TSV" ] || printf "quad\tarm\tattempt\trc\tstate\tstalled\tstart\tend\n" > "$TSV"
echo "=== LOAM RUN 3: quads $FIRST_QUAD..$((FIRST_QUAD + QUADS - 1)), output $DATA_DIR, retries $MAX_RETRIES ==="
echo "Started: $(date)"

state_of() { "$PYTHON" "$STATE" "$1"; }

# activity = transcript size + event-log mtime; unchanged for STALL_S means stuck
if stat --version >/dev/null 2>&1; then SIZE="stat -c %s"; MTIME="stat -c %Y"; else SIZE="stat -f %z"; MTIME="stat -f %m"; fi
activity() {
  local d=$1
  echo "$($SIZE "$d/transcript.jsonl" 2>/dev/null || echo 0)-$($MTIME "$d/event_log.json" 2>/dev/null || echo 0)"
}

run_session() {  # quad arm -> leaves a final state in $arm_dir
  local q=$1 arm=$2
  local qd; qd=$(printf '%02d' "$q")
  local arm_dir="$DATA_DIR/quad_$qd/$arm"
  local attempt st
  st=$(state_of "$arm_dir")
  case "$st" in COMPLETE|FINAL_*) echo "  $arm: already $st"; return 0 ;; esac
  # attempts already made (a restarted runner continues the count)
  attempt=$(ls -d "$arm_dir".attempt* 2>/dev/null | wc -l | tr -d ' ')
  while :; do
    if [ "$st" = "TECHNICAL" ]; then          # a failed attempt is in place: keep it aside and count it
      mv "$arm_dir" "$arm_dir.attempt$attempt.technical"
      attempt=$((attempt + 1))
    fi
    if [ "$attempt" -gt "$MAX_RETRIES" ]; then
      echo "  $arm: TECHNICAL after $attempt attempts; giving up"
      return 1
    fi
    local log="$DATA_DIR/logs/quad_${qd}_${arm}_attempt${attempt}.log"
    local start; start=$(date '+%F %T')
    "$PYTHON" "$SCRIPT" --arm "$arm" --quad "$q" --seed "$SEED" --data-dir "$DATA_DIR" > "$log" 2>&1 < /dev/null &
    local pid=$! last="" same=0 stalled=no
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL_S"
      local now; now=$(activity "$arm_dir")
      if [ "$now" = "$last" ]; then same=$((same + POLL_S)); else same=0; last=$now; fi
      if [ "$same" -ge "$STALL_S" ]; then
        stalled=yes
        kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null
        break
      fi
    done
    wait "$pid" 2>/dev/null; local rc=$?
    st=$(state_of "$arm_dir")
    if [ "$st" = "NONE" ]; then               # died before writing anything: still a technical stop
      mkdir -p "$arm_dir"; st=TECHNICAL
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$q" "$arm" "$attempt" "$rc" "$st" "$stalled" "$start" "$(date '+%F %T')" >> "$TSV"
    echo "  $arm attempt $attempt: rc=$rc state=$st stalled=$stalled"
    tail -3 "$log" | sed 's/^/    | /'
    case "$st" in COMPLETE|FINAL_*) return 0 ;; esac
    sleep "$CLEANUP_S"   # MPS cleanup
  done
}

for q in $(seq "$FIRST_QUAD" $((FIRST_QUAD + QUADS - 1))); do
  echo ""
  echo "--- Quad $q ($(date)) ---"
  for arm in $ARMS; do
    if [ "$arm" = "observed" ]; then
      est=$(state_of "$DATA_DIR/quad_$(printf '%02d' "$q")/enacted")
      case "$est" in
        COMPLETE|FINAL_*) ;;
        *) echo "  observed: skipped, enacted is $est"
           printf "%s\tobserved\t-\t-\tSKIPPED_ENACTED_%s\t-\t%s\t-\n" "$q" "$est" "$(date '+%F %T')" >> "$TSV"
           continue ;;
      esac
    fi
    run_session "$q" "$arm"
    sleep "$CLEANUP_S"   # MPS cleanup
  done
done

echo ""
echo "=== LOAM RUN 3 COMPLETE: $(date) ==="
cut -f2,5 "$TSV" | tail -n +2 | sort | uniq -c
