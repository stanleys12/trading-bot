#!/bin/zsh
# The wall: 20 tiles, one per agent, latest memo in each. Session: fund20
TMUX_BIN=$(command -v tmux || echo $HOME/.local/bin/tmux)
R=$(cd "$(dirname "$0")/.." && pwd) || exit 1
PY=${PY:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3}
S=fund20
"$TMUX_BIN" has-session -t ${S} 2>/dev/null && { [ -t 0 ] && exec "$TMUX_BIN" attach -t ${S} || { echo "already running; attach with: tmux attach -t fund20"; exit 0; }; }
SEATS=(chief_of_desk risk_officer execution_verifier measurement_auditor ops_engineer equity_pm crypto_pm options_pm scout_equity scout_crypto scout_macro news_analyst research_lead quant_a quant_b red_team backtest_auditor evals_steward coach scribe)
"$TMUX_BIN" new-session -d -s ${S} -n wall -x 240 -y 60 -c "$R"
"$TMUX_BIN" set-option -t ${S} -g automatic-rename off
"$TMUX_BIN" set-option -t ${S} -g pane-border-status top
"$TMUX_BIN" set-option -t ${S} -g pane-border-format " #{@title} "
i=0
for seat in $SEATS; do
  if [ $i -gt 0 ]; then
    "$TMUX_BIN" split-window -t ${S}:wall -c "$R" 2>/dev/null
    "$TMUX_BIN" select-layout -t ${S}:wall tiled
  fi
  "$TMUX_BIN" set-option -p -t ${S}:wall.$i @title "$seat"
  "$TMUX_BIN" send-keys -t ${S}:wall.$i "while true; do clear; $PY fund/seat_view.py $seat 2>/dev/null | head -12; sleep 4; done" C-m
  i=$((i+1))
done
for extra in "★ DESK NOTE|while true; do clear; cat fund/reports/desk_note_latest.md 2>/dev/null | head -12; sleep 30; done" "★ SPRINT VERDICTS|while true; do clear; $PY fund/sprint_status.py 2>/dev/null | head -12; sleep 30; done" "★ DECISION JOURNAL|tail -f fund/journal/journal.jsonl" "★ LIVE FLOOR FEED|tail -n 8 -f fund/reports/floor_feed.log"; do
  title="${extra%%|*}"; cmd="${extra#*|}"
  "$TMUX_BIN" split-window -t ${S}:wall -c "$R" 2>/dev/null
  "$TMUX_BIN" select-layout -t ${S}:wall tiled
  idx=$("$TMUX_BIN" list-panes -t ${S}:wall -F "#{pane_index}" | tail -1)
  "$TMUX_BIN" set-option -p -t ${S}:wall.$idx @title "$title"
  "$TMUX_BIN" send-keys -t ${S}:wall.$idx "$cmd" C-m
done
"$TMUX_BIN" select-layout -t ${S}:wall tiled
[ -t 0 ] && exec "$TMUX_BIN" attach -t ${S} || echo "agent wall started; attach with: tmux attach -t fund20"
