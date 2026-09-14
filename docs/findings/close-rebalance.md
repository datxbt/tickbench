# Rebalancing flow into the US close - evaluation

Subject: leveraged and inverse index funds must trade with the day's move near
the close - a 3x fund buys six times its AUM times the day's return - and short-
gamma dealers hedge the same way (Cheng & Madhavan 2009; Baltussen, Da, Lammers
& Martens 2021). The Nasdaq-100 carries the largest such complex. The
prediction: the last half hour continues the day's move, in proportion to it.

USTEC's rejection log tested first-30-minutes -> last-30 (corr -0.047), which is
not this predictor. This study is adjacent to the Zarattini intraday-momentum
family ([intraday-momentum-and-btc-dynamics.md](intraday-momentum-and-btc-dynamics.md)),
whose test split is spent, and nothing here goes near that split.

Method: `scripts/research/close_rebalance.py`. Signal is the sign of the mid
return from the prior session's 16:00 ET close to 15:30 ET; trade in that
direction 15:30 -> 16:00 at one-minute bar-close quotes, real fills, commission
and measured slippage. Dev and validation, pooled.

**Verdict: rejected. The pre-registered primary cell is -0.13 bps (t -0.11), and
validation goes significantly the wrong way. The test split was not spent.**

| cell | n | net bps | t | dev | validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| **USTEC, all days, 15:30** | 1,327 | **-0.13** | -0.11 | +1.42 (t +0.93) | -4.28 (t -2.64) |
| USTEC, \|day\| > 1% | 577 | -0.30 | -0.13 | +1.28 | -5.63 (t -1.93) |
| USTEC, all days, 15:45 | 1,326 | -0.09 | -0.09 | +0.75 | -2.35 |
| XAUUSD, all days | 1,336 | +0.10 | +0.30 | +0.41 | -0.72 |
| EURUSD, all days | 1,347 | -0.41 | -2.96 | -0.21 | -0.93 |
| USDJPY, all days | 1,347 | -0.68 | -4.77 | -0.71 | -0.59 |

The dose-response the mechanism needs - last-half-hour return regressed on the
day's return - is +0.0175 at t = +1.03 on USTEC: the right sign, a slope of under
two basis points per 1% move, and not significant. The FX pairs simply pay their
round turn; their mids are flat.

Gold is the one instrument with a significant slope (+0.0113, t = +2.72), and it
is the one with no reason to have it: gold is a falsification instrument here,
with no leveraged-fund complex rebalancing at the US equity close. Recorded, not
pursued - chasing a pattern on the control instrument after the primary failed
is the selection this project's protocol exists to prevent.

```bash
python scripts/research/close_rebalance.py
```
