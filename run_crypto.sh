#!/bin/bash
# Crypto TSMOM paper forward test (frozen spec exp14, hash f3a5afb0).
# PAPER MONEY ONLY. Daily at 06:15 PT, seven days a week -- crypto has no
# session close and the 7-day evaluation grid can land on a weekend.
# Same retry rationale as run_daily.sh: the runner reconciles positions to
# the grid target, so re-running after a transient failure is safe.
cd "$(dirname "$0")" || exit 1

PY=${PY:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3}
LOG=qr/reports/live/crypto_cron.log

for attempt in 1 2 3; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt}" >> "$LOG"
  "$PY" -W ignore -m qr.execution.crypto_paper run >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt} succeeded" >> "$LOG"
    exit 0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt} failed (rc=${rc})" >> "$LOG"
  [ "$attempt" -lt 3 ] && sleep 600
done
echo "$(date '+%Y-%m-%d %H:%M:%S') all attempts failed" >> "$LOG"
exit 1
