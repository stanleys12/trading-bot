# trading-bot

Quantitative trading research, mostly US small-cap equities and later crypto, run as pre-registered experiments with a sealed test set.

The short version: 212 pre-registered trials, none promoted, the sealed test set never opened. The goal was to find out honestly whether a small, real edge exists after costs, liquidity limits and bias controls, not to produce the best-looking backtest. The full write-up is in [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Layout

- `qr/` research framework: data pipeline, backtesting, strategies, portfolio, paper execution (live trading is blocked in code), and the `sprint/` intraday crypto experiments with their specs and rejected proposals
- `fund/` agent-run paper fund: orchestrator, dispatcher, playbooks
- `src/`, `research/` earlier experiments
- `run_*.sh` entry points used by the schedulers

Market data, generated reports, trade journals and results are not in this repo. Paper broker keys go in `.env` (see the scripts), never in code.

Built with heavy use of Claude Code.
