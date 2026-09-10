# Big bar, pause bar, break of the pause bar - evaluation

Subject: the price-action classic, stated in four steps.

1. A big bar breaks out of resistance.
2. The next bar is a very small pause bar.
3. Enter on the break of the pause bar.
4. Stop under it.

With the claim attached: *works on all assets and timeframes.*

Method: implemented in `src/qlab/strategies/pause_bar.py` and run tick-by-tick
over the whole 2020-2025 corpus on all four instruments at four timeframes
spanning 60x - 1m, 5m, 15m, 1h. **149,473 signals, 77,432 filled trades.**
Fills cross a real bid and a real ask from the tape, so spread is observed
rather than assumed; commission is the published contract term; slippage is
`qlab.costs.CostModel` at the default 250 ms / 0.5 adverse fraction, charged on
market fills - the entry, a stop-out, a trail-out, a time exit - and never on a
fixed target, which is a limit and fills at its price or not at all.

Steps 1-4 fix the entry and the risk and say nothing about the way out, so no
single exit is assumed. Nine are resolved against the same fills: targets at
0.5R, 1R, 1.5R, 2R, 3R, 4R and 6R, a breakeven-then-trail, and a plain time
stop. Everything is reported in **R**, multiples of the planned risk from
trigger to stop - the only unit in which a 1.1-pip EURUSD stop and a 467-pip
gold stop are the same bet.

**Verdict: do not trade this. Rejected on the dev split; the held-out test
split was not spent.** The setup loses on every instrument at every timeframe
except one 672-trade cell that does not replicate. Of 160 (instrument x
timeframe x exit) cells on dev, **zero** clear t = +2 and 87 clear t = -2. The
loss is not bad luck and it is not a bad exit: it is arithmetic, and the
arithmetic is a direct consequence of step 2.

---

## 1. Step 2 sets the stop, and the stop sets the cost

The pause bar is selected *for being small*. The stop goes under it. So the
smaller the pause bar - the better the setup looks by its own criteria - the
smaller 1R is, and the larger a fixed round-turn cost becomes as a fraction of
it.

Round-turn cost as a fraction of the amount risked, median over dev:

| | 1m | 5m | 15m | 1h |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | 0.532 | 0.203 | 0.102 | 0.043 |
| USDJPY | 0.613 | 0.228 | 0.115 | 0.049 |
| XAUUSD | 0.657 | 0.200 | 0.108 | 0.040 |
| USTEC | 0.515 | 0.174 | 0.085 | 0.032 |

On one-minute bars the median trade pays **half to two-thirds of everything it
risks** just to get in and out. The median 1m risk is 1.0-2.1 bps of price;
the round turn on this account is 0.5-0.7 bps. Those two numbers are the same
size, and that is the whole story.

Put as a break-even win rate on a 2R target, where a winner pays +2R and a
loser -1R:

| timeframe | needs to win | actually wins | shortfall |
| --- | ---: | ---: | ---: |
| 1m | 52.7% | 31.7% | -21.0 pts |
| 5m | 40.0% | 33.3% | -6.7 pts |
| 15m | 36.7% | 31.9% | -4.8 pts |
| 1h | 34.7% | 28.4% | -6.3 pts |

The gap narrows as the timeframe lengthens, but it never closes, and the reason
it narrows is that the stop gets wider - which is the opposite of step 2.

## 2. The loss is the cost, to two decimal places

Pooled over the four instruments on dev, at a 2R target: the same trades priced
at mid, less the measured cost, reproduces the realised result to within 0.02R
at every timeframe.

| timeframe | at mid | cost | mid - cost | actual net | residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1m | +0.126 | 0.811 | -0.685 | **-0.706** | -0.022 |
| 5m | +0.071 | 0.251 | -0.180 | **-0.191** | -0.012 |
| 15m | +0.043 | 0.121 | -0.078 | **-0.087** | -0.009 |
| 1h | +0.084 | 0.048 | +0.036 | **+0.022** | -0.013 |

There is no execution subtlety left to find here and no better broker to go
looking for. The strategy is short a fixed toll and long nothing much.

## 3. There is no edge before costs either

The table above shows a small positive number in the "at mid" column, and that
is the last place this idea could have been alive: an edge too small for
*these* costs might still be worth something on a cheaper venue.

It is not an edge. Fading every signal - selling the break of the pause bar's
low on a bullish setup, buying its high on a bearish one, same geometry
mirrored - is **also** positive at mid, by as much or more:

| timeframe | signal, at mid | faded, at mid | sum |
| --- | ---: | ---: | ---: |
| 1m | +0.126 | +0.153 | +0.279 |
| 5m | +0.071 | +0.026 | +0.098 |
| 15m | +0.043 | +0.047 | +0.090 |
| 1h | +0.084 | +0.091 | +0.174 |

A real edge reverses sign when you fade it, so these should sum to about zero.
They sum to roughly twice themselves. What the "at mid" column measures is a
property of the *measurement*: pricing a stop-and-target trade at mid, when its
entry is a stop order crossing the spread and its exits are triggered on the
far side of the book, hands back spread and entry slippage on every trade, win
or lose. Mean `mid - gross` is +0.377R on 1m, +0.111R on 5m, +0.056R on 15m and
+0.030R on 1h - the same shape, and the same order of size, as the numbers in
the table. It is paid to the signal and to its mirror alike.

The measure with no barrier and no fill price in it agrees. Unconditional
forward return from the entry mid over the holding window, in basis points,
aggregated daily:

| | dev mean/day | t | validation mean/day | t |
| --- | ---: | ---: | ---: | ---: |
| 1m | -0.23 | **-2.56** | +0.12 | 0.82 |
| 5m | -0.22 | -0.45 | -0.37 | -0.61 |
| 15m | -1.67 | -1.44 | -1.76 | -1.28 |
| 1h | +3.51 | 1.13 | +9.11 | 1.81 |

Eight cells, no consistent sign, one significant - and it is negative.

## 4. Step 1 does no work

If the breakout context matters, then keeping steps 2-4 and throwing step 1
away should cost something measurable. It does not. Same entry geometry - a
small bar after a directional bar, entered on its break - with the "big" test
and the resistance test both dropped:

| context | trades | cost/R | forward R | t | 2R at mid |
| --- | ---: | ---: | ---: | ---: | ---: |
| breakout (the setup) | 11,275 | 0.161 | +0.074 | 1.34 | +0.065 |
| explicitly *not* a breakout | 18,591 | 0.199 | -0.012 | -0.29 | +0.039 |
| geometry only, no context | 77,069 | 0.249 | +0.045 | 1.65 | **+0.072** |

The geometry-only control has a *higher* cost-free result than the setup, on
seven times the sample. Per timeframe the setup's advantage over it is -0.015R
at 5m, +0.010R at 15m and +0.078R at 1h - the last on 672 trades against 3,651.
The breakout condition is a way of trading less often, not of trading better.

## 5. All assets, all timeframes

Mean R net of all costs, dev, at the exit that was best for that cell - which
is already an in-sample choice in the idea's favour:

| | 1m | 5m | 15m | 1h |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | -0.430 | -0.073 | -0.070 | +0.153 |
| USDJPY | -0.418 | +0.002 | -0.020 | -0.005 |
| XAUUSD | -0.827 | -0.279 | -0.092 | +0.059 |
| USTEC | -0.735 | -0.168 | -0.024 | +0.286 |

Pooled, with the plain 2R target the idea most often comes with:

| split | tf | signals | filled | 2R net | daily t | best exit | best R |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| dev | 1m | 95,875 | 50,336 | -0.706 | **-41.4** | trail | -0.616 |
| dev | 5m | 15,574 | 7,897 | -0.191 | **-11.0** | trail | -0.150 |
| dev | 15m | 5,363 | 2,706 | -0.087 | **-3.1** | 3R | -0.081 |
| dev | 1h | 1,367 | 672 | +0.022 | 0.4 | 4R | +0.045 |
| val | 1m | 25,114 | 12,678 | -0.495 | **-25.7** | trail | -0.480 |
| val | 5m | 4,193 | 2,143 | -0.195 | **-5.9** | 1R | -0.176 |
| val | 15m | 1,551 | 798 | -0.106 | **-2.1** | 4R | -0.090 |
| val | 1h | 436 | 202 | +0.058 | 0.6 | time stop | +0.143 |

Only the 1h row is positive, on 672 dev trades and 202 validation trades - and
it is not significant in either, which is what a mean of +0.02R on that sample
size means. Taking the best exit per instrument on dev and carrying it forward
unchanged, 8 of 12 instrument-timeframe cells lose on validation and none is
significant either way.

The claim "works on all assets and timeframes" is testable and it is false in
the strong direction: the results are consistent across instruments, and
consistently negative.

## 6. No parameter setting rescues it

Each parameter moved on its own, on dev, at 15m and 1h - the timeframes where
the cost objection is weakest, so the sweep is looking where the edge has its
best chance:

| parameter | values tried | best 2R | best of any exit |
| --- | --- | ---: | ---: |
| `big_mult` | 1.0 - 2.5 | +0.001 | +0.001 |
| `pause_mult` | 0.25 - 0.7 | -0.016 | -0.004 |
| `close_frac` | 0.5 - 0.9 | -0.037 | +0.101 |
| `lookback` | 10 - 80 | -0.058 | -0.049 |
| `arm_bars` | 1 - 3 | -0.065 | -0.059 |
| `max_hold_bars` | 5 - 40 | -0.067 | -0.041 |
| `require_inside` | off / on | -0.066 | -0.059 |

Twenty-six settings, ~230 (setting x exit) combinations, one positive result
above +0.05R. That is fewer than chance would give.

Two of these deserve a note, because they point the same way:

- **Making the setup stricter makes it worse.** Demanding a true inside bar -
  the textbook version of step 2 - takes mean R from -0.066 to **-0.149** and
  the sample from 3,378 to 1,404. Tightening `pause_mult` to 0.25, the most
  faithful reading of "very small", leaves 188 trades in four years.
- **What helps is a bigger stop.** `big_mult=2.5` is the only setting to reach
  zero, and it does so by selecting breakout bars large enough that the pause
  bar after them is not that small. Within any timeframe, sorting trades by
  their own risk says the same thing:

  | 15m, dev | cost/R | 2R at mid | 2R net |
  | --- | ---: | ---: | ---: |
  | tightest fifth | 0.197 | +0.106 | -0.121 |
  | widest fifth | 0.047 | +0.072 | **+0.015** |

  The cost-free column is flat across the quintiles; only the cost moves. The
  cure for this strategy is to stop choosing small pause bars - which is to
  stop trading the setup.

## 7. Why the tight stop backfires twice

The appeal of step 2 is a small 1R and therefore a large reward-to-risk ratio.
Both halves of that are worse than they look.

**The fill is past the trigger.** A stop order is filled at the ask on a rising
market, and the tighter the level the bigger the overshoot relative to it. On
1m the realised risk is a median **1.089x** the planned risk before any cost;
by 1h it is 1.007x.

**A loss is not 1R.** Adding the overshoot to the round turn, the average
losing trade at a 2R target gives back:

| timeframe | average loss | on a stop-out |
| --- | ---: | ---: |
| 1m | -1.76R | -1.77R |
| 5m | -1.24R | -1.24R |
| 15m | -1.11R | -1.13R |
| 1h | -1.00R | -1.06R |

So the 1m version of this setup is not risking 1 to make 2. It is risking 1.76
to make 1.54 - the average winner comes in under the nominal target too,
because time exits are counted in - at a 32% win rate. The reward-to-risk ratio
the setup is sold on is a bar-chart property that the tape does not honour.

## 8. What this does not say

- **It is not a claim that price-action reading is worthless.** It is a
  measurement of one mechanical transcription of one setup, on one broker's
  cost structure, over 2020-2025. A discretionary trader filtering these
  signals by context this code cannot see is not covered by it.
- **The cost model is not the reason.** Set slippage to zero entirely - a claim
  no chasing stop-order entry can make - and 1m still pays 0.50R a round turn
  in spread and commission alone (0.17R at 5m, 0.09R at 15m, 0.03R at 1h).
- **The test split was not spent.** Nothing reached it: the rule in this project
  is that `test` buys one honest estimate for a candidate that survived dev and
  validation, and this candidate survived neither. `SPLITS["test"]` remains
  unspent for the next idea.

## 9. If you want to keep pulling this thread

In rough order of how much the evidence above supports them:

1. **Trade the same idea where 1R is 20+ bps, not 2.** Everything here is
   consistent with a setup that is fine in principle and uneconomic at the risk
   sizes step 2 produces. 4h and daily bars on these instruments would put
   cost/R near 0.01. The sample gets small fast - 1h already gives only 672 dev
   trades across four instruments - so this is a multi-year-of-data question,
   not a weekend one.
2. **Keep the pause bar, drop the stop.** The entry geometry is not the problem;
   the risk denominator is. A fixed ATR stop with the pause-bar *trigger* would
   separate the two, and the trigger is the half of the idea that at least does
   no harm.
3. **Stop looking for the exit.** Nine exits were resolved on identical fills
   and they span 0.09R at 5m and 0.13R at 15m - less than the cost at either.
   The exit is not where this is decided, and a search over exits on a signal
   with no measurable edge is a search over noise.

## 10. Reproducing

```bash
python scripts/backtests/backtest_pause_bar.py              # dev + validation, the tables above
python scripts/backtests/backtest_pause_bar.py --control    # the geometry-only null
python scripts/backtests/backtest_pause_bar.py --placebo    # the faded-signal symmetry check
python scripts/backtests/backtest_pause_bar.py --sweep      # one-at-a-time parameter sensitivity
python -m pytest tests/test_pause_bar.py          # fill, ordering and detection logic
```

Trade tapes land in `reports/strategies/pause_bar_trades/`, one parquet per
split, carrying entry and exit mids alongside the fills so that "the signal had
no edge" and "the edge existed and execution ate it" stay separable - here it
was neither and both, and only a tape with both prices on it can say so.

The whole study runs in about two minutes on this corpus. Signals are detected
once per timeframe on bars and then resolved in a single pass over each month of
tape, so the four timeframes cost one walk through the ticks rather than four.
