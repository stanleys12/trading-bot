#!/bin/zsh
# Resume every trading-bot background job.
U=$(id -u)
for l in com.stanleyshen.tradingbot.daily com.stanleyshen.tradingbot.crypto com.stanleyshen.tradingbot.options com.stanleyshen.tradingfund com.stanleyshen.dispatcher com.stanleyshen.firmlog com.stanleyshen.actionslog com.stanleyshen.pmrecorder com.stanleyshen.weatherrec com.stanleyshen.weatherpaper com.stanleyshen.ensemble; do
  launchctl enable gui/$U/$l
  launchctl bootstrap gui/$U ~/Library/LaunchAgents/$l.plist 2>/dev/null
done
rm -f "$(dirname "$0")"/PAUSED
echo "trading bot resumed $(date)"
