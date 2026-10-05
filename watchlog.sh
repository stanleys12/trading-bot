#!/bin/zsh
# THE FIRM LOG — live view. Run this in any terminal. Ctrl-C to stop watching
# (the collector keeps running in the background either way).
exec tail -n 40 -f /Users/stanleyshen/trading-bot/fund/reports/firm.log
