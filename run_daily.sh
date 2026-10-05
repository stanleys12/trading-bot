#!/bin/bash
# Daily paper-trading forward test. PAPER MONEY ONLY.
# Scheduled for 09:40 ET (10 min after the open, so quotes have settled).
#
# Retry loop: the 2026-08-13 scheduled run died on a transient data-API
# timeout and lost the whole trading day, because launchd would not try again
# until the next weekday. The runner is idempotent (orders are deltas vs
# current positions) and exits 0 both on success and on "market closed", so
# re-running after a failure is safe by construction. Four attempts, ten
# minutes apart, covers ~40 minutes of API flakiness without extending past
# the trading morning.
cd /Users/stanleyshen/trading-bot

PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
LOG=qr/reports/live/cron.log

for attempt in 1 2 3 4; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt}" >> "$LOG"
  "$PY" -W ignore -m qr.execution.daily_run >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt} succeeded" >> "$LOG"
    exit 0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt ${attempt} failed (rc=${rc})" >> "$LOG"
  [ "$attempt" -lt 4 ] && sleep 600
done
echo "$(date '+%Y-%m-%d %H:%M:%S') all attempts failed" >> "$LOG"
exit 1
