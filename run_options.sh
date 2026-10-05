#!/bin/bash
# Covered-call paper book (vol risk premium forward test). PAPER MONEY ONLY.
# 06:50 PT weekdays -- after the open, after the crypto/equity runs.
# Same retry rationale as the other runners: the book reconciles state
# (shares held, calls short, resting orders) so re-running is safe.
cd /Users/stanleyshen/trading-bot

PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
LOG=qr/reports/live/options_cron.log

for attempt in 1 2 3; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt}" >> "$LOG"
  "$PY" -W ignore -m qr.execution.options_paper run >> "$LOG" 2>&1
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
