#!/bin/zsh
# Single-screen mission control: everything in ONE window, six panes.
#   fund/dashboard1.sh     start or reattach   (session: fund1)
# Layout:
#   left           | middle              | right
#   desk note      | sprint verdicts     | orchestrator log
#   talk shell     | all 3 book logs     | decision journal
TMUX_BIN=$(command -v tmux || echo $HOME/.local/bin/tmux)
R=/Users/stanleyshen/trading-bot
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
S=fund1
"$TMUX_BIN" has-session -t ${S} 2>/dev/null && { [ -t 0 ] && exec "$TMUX_BIN" attach -t ${S} || { echo "already running; attach with: tmux attach -t fund1"; exit 0; }; }

"$TMUX_BIN" new-session -d -s ${S} -n all -c "$R"
"$TMUX_BIN" set-option -t ${S} -g automatic-rename off
"$TMUX_BIN" set-option -t ${S} -g pane-border-status top
"$TMUX_BIN" set-option -t ${S} -g pane-border-format " #{@title} "

# three columns
"$TMUX_BIN" split-window  -h -l 66% -t ${S}:all.0 -c "$R"     # -> pane 1
"$TMUX_BIN" split-window  -h -l 50% -t ${S}:all.1 -c "$R"     # -> pane 2
# split each column
"$TMUX_BIN" split-window  -v -l 30% -t ${S}:all.0 -c "$R"     # -> pane 3 (left-bottom)
"$TMUX_BIN" split-window  -v -l 45% -t ${S}:all.1 -c "$R"     # -> pane 4 (mid-bottom)
"$TMUX_BIN" split-window  -v -l 50% -t ${S}:all.2 -c "$R"     # -> pane 5 (right-bottom)

"$TMUX_BIN" split-window -v -b -l 13 -t ${S}:all.0 -c "$R"
"$TMUX_BIN" set-option -p -t ${S}:all.0 @title "ALL 20 AGENTS"
"$TMUX_BIN" send-keys  -t ${S}:all.0 "while true; do clear; $PY fund/agents_status.py; sleep 20; done" C-m
"$TMUX_BIN" set-option -p -t ${S}:all.1 @title "DESK NOTE"
"$TMUX_BIN" send-keys  -t ${S}:all.1 "while true; do clear; cat fund/reports/desk_note_latest.md 2>/dev/null | head -50; sleep 30; done" C-m
"$TMUX_BIN" set-option -p -t ${S}:all.3 @title "TALK (fund/talk.py <seat>)"
"$TMUX_BIN" send-keys  -t ${S}:all.3 "clear; $PY fund/talk.py" C-m
"$TMUX_BIN" set-option -p -t ${S}:all.1 @title "SPRINT VERDICTS"
"$TMUX_BIN" send-keys  -t ${S}:all.1 "while true; do clear; $PY fund/sprint_status.py; sleep 30; done" C-m
"$TMUX_BIN" set-option -p -t ${S}:all.4 @title "TRADING BOOKS (x3)"
"$TMUX_BIN" send-keys  -t ${S}:all.4 "tail -n 3 -f qr/reports/live/cron.log qr/reports/live/crypto_cron.log qr/reports/live/options_cron.log" C-m
"$TMUX_BIN" set-option -p -t ${S}:all.2 @title "ORCHESTRATOR"
"$TMUX_BIN" send-keys  -t ${S}:all.2 "tail -n 20 -f fund/reports/fund_cron.log" C-m
"$TMUX_BIN" set-option -p -t ${S}:all.5 @title "DECISION JOURNAL"
"$TMUX_BIN" send-keys  -t ${S}:all.5 "tail -n 5 -f fund/journal/journal.jsonl" C-m

[ -t 0 ] && exec "$TMUX_BIN" attach -t ${S} || echo "single-screen dashboard started; attach with: tmux attach -t fund1"
