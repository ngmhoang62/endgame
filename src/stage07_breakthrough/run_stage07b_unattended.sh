#!/usr/bin/env bash
set -u

DRIVE_ROOT="${1:-/content/drive/MyDrive/DSC2026/stage07b}"
LOG_DIR="$DRIVE_ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/run_${STAMP}.log"

echo "[stage07b] unattended run; log=$LOG" | tee -a "$LOG"

STATUS=0
python /content/stage07b/run_qwen4b_teacher_portable_a100.py --drive-root "$DRIVE_ROOT" 2>&1 | tee -a "$LOG"
T=${PIPESTATUS[0]}
if [ "$T" -ne 0 ]; then
  echo "[stage07b] teacher failed status=$T" | tee -a "$LOG"
  STATUS=$T
else
  python /content/stage07b/run_qwen06b_student_portable_a100.py --drive-root "$DRIVE_ROOT" 2>&1 | tee -a "$LOG"
  S=${PIPESTATUS[0]}
  if [ "$S" -ne 0 ]; then
    echo "[stage07b] student failed status=$S" | tee -a "$LOG"
    STATUS=$S
  fi
fi

echo "[stage07b] finished status=$STATUS; shutting down runtime to stop compute-unit burn." | tee -a "$LOG"
python /content/stage07b/shutdown_colab_runtime.py
exit "$STATUS"
