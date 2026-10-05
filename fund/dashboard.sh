#!/bin/zsh
# Mission control for the AI fund desk, in tmux.
#   fund/dashboard.sh          start (or reattach to) the dashboard
# Detach: Ctrl-b then d   |  Switch window: Ctrl-b then 0/1/2
# Switch pane: Ctrl-b then arrow key   |  Scroll: Ctrl-b then [ (q to exit)
TMUX_BIN=$(command -v tmux || echo $HOME/.local/bin/tmux)
if [ ! -x "$TMUX_BIN" ] && [ ! -x /usr/local/bin/tmux ]; then
  echo "tmux is not installed yet. Install Homebrew + tmux with:"
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  echo '  eval "$(/opt/homebrew/bin/brew shellenv)" && brew install tmux'
  echo "then run fund/dashboard.sh again."
  exit 1
fi
[ -x "$TMUX_BIN" ] || TMUX_BIN=/usr/local/bin/tmux
R=/Users/stanleyshen/trading-bot
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
S=fund
"$TMUX_BIN" has-session -t ${S} 2>/dev/null && { [ -t 0 ] && exec "$TMUX_BIN" attach -t ${S} || { echo "session '${S}' already running; attach with: tmux attach -t fund"; exit 0; }; }

# window 0: DESK — desk note | live alerts | talk shell
"$TMUX_BIN" new-session -d -s $S -n desk -c "$R"
"$TMUX_BIN" set-option -t ${S} -g automatic-rename off
"$TMUX_BIN" set-option -t ${S} -g allow-rename off
"$TMUX_BIN" send-keys -t ${S}:desk "while true; do clear; cat fund/reports/desk_note_latest.md 2>/dev/null | head -40; sleep 30; done" C-m
"$TMUX_BIN" split-window -h -t ${S}:desk -c "$R"
"$TMUX_BIN" send-keys -t ${S}:desk "tail -f fund/reports/fund_cron.log" C-m
"$TMUX_BIN" split-window -v -t ${S}:desk -c "$R"
"$TMUX_BIN" send-keys -t ${S}:desk "clear; echo 'TALK TO A SEAT:'; echo '  $PY fund/talk.py                    (roster)'; echo '  $PY fund/talk.py risk_officer       (chat)'" C-m

# window 1: BOOKS — the three trading books' live logs
"$TMUX_BIN" new-window -t $S -n books -c "$R"
"$TMUX_BIN" send-keys -t ${S}:books "tail -f qr/reports/live/cron.log" C-m
"$TMUX_BIN" split-window -h -t ${S}:books -c "$R"
"$TMUX_BIN" send-keys -t ${S}:books "tail -f qr/reports/live/crypto_cron.log" C-m
"$TMUX_BIN" split-window -v -t ${S}:books -c "$R"
"$TMUX_BIN" send-keys -t ${S}:books "tail -f qr/reports/live/options_cron.log" C-m

# window 2: SPRINT — experiments, verdicts, registry
"$TMUX_BIN" new-window -t $S -n sprint -c "$R"
"$TMUX_BIN" send-keys -t ${S}:sprint "while true; do clear; $PY fund/sprint_status.py; sleep 30; done" C-m
"$TMUX_BIN" split-window -v -t ${S}:sprint -c "$R"
"$TMUX_BIN" send-keys -t ${S}:sprint "tail -f fund/journal/journal.jsonl" C-m
[ -t 0 ] && exec "$TMUX_BIN" attach -t ${S} || echo "dashboard started; attach with: tmux attach -t fund"
