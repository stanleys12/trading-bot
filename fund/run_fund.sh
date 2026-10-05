#!/bin/zsh
# Fund desk cycle. launchd fires hourly; orchestrator.py decides whether this
# hour is a full desk, a light scan, skeleton crew, research, or a skip.
# Python is pinned: launchd PATH resolves python3 to the system build
# without site-packages (the lesson from the fantasy autopilot incident).
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
cd /Users/stanleyshen/trading-bot || exit 1
FUND_WORKERS=${FUND_WORKERS:-5} "$PY" -W ignore fund/orchestrator.py >> fund/reports/fund_cron.log 2>&1
