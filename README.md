# gta6-twitch-ccu-forecast-01

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/indrairawan4503-cmd/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/indrairawan4503-cmd/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), after the first hour.

## What it does

`main.py` (Python 3.13, standard library only) reads a case file and
writes five quantiles (5% / 25% / 50% / 75% / 95%) of Twitch viewers
for hours 2–24 of release day, anchored to the case's hour-1 print.

The model is an empirical comparable-shape model: for each past big
launch it takes the log-ratio of each hour's viewers to that launch's
hour-1 viewers, then per hour regresses that log-ratio on
log(hour-1 viewers) across launches — big first hours are already near
peak, small ones are still climbing into their unlock waves — with the
slope shrunk 20% toward zero. Console-only rolling-midnight launches
are weighted highest (GTA VI is console-only, midnight local unlock);
launches whose first hour was under 10% of their own day peak
(unlock had not happened yet) are excluded, and PC-simultaneous
launches are downweighted. The spread at each hour is the weighted
residual standard deviation across launches, so the 5–95% band is the
honest cross-launch spread, not a guessed width.

Validated by leave-one-out over the comparables: each past launch's
release day is treated as a pseudo-case (hour-1 print known, hours
2–24 forecast) and scored with the contest's exact pinball loss on the
log scale. Mean loss ≈ 0.17–0.20 on console-only rolling launches vs
≈ 0.32 for the house baseline on the same pseudo-cases.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Referral

Built for the make.fun bounty — join with my referral link:
https://make.fun/r/8dxq9bq0
