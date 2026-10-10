#!/bin/zsh
# THE ACTIONS LOG — plain-English trades & changes only. Ctrl-C to stop watching.
exec tail -n 30 -f "$(dirname "$0")"/fund/reports/actions.log
