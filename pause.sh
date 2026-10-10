#!/bin/zsh
# Pause every trading-bot background job. Survives reboot. Undo with ./resume.sh
U=$(id -u)
for l in com.stanleyshen.tradingbot.daily com.stanleyshen.tradingbot.crypto com.stanleyshen.tradingbot.options com.stanleyshen.tradingfund com.stanleyshen.dispatcher com.stanleyshen.firmlog com.stanleyshen.actionslog com.stanleyshen.pmrecorder com.stanleyshen.weatherrec com.stanleyshen.weatherpaper com.stanleyshen.ensemble; do
  launchctl disable gui/$U/$l
  launchctl bootout gui/$U/$l 2>/dev/null
done
echo "trading bot paused $(date)" | tee -a "$(dirname "$0")"/PAUSED
