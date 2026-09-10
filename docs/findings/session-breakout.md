# Session opening-range breakout - evaluation

Subject: `XAUUSD_SessionBreakout_2026.mq5`, v2.00, preset `GEO_2026_NO_H13`.

Method: the expert was re-implemented tick-by-tick in
`src/qlab/strategies/session_breakout.py` and run over the full 2020-2026 corpus
on all four instruments. Fills cross a real bid and a real ask from the tape, so
spread is observed rather than assumed; commission is the published contract
term; slippage is `qlab.costs.CostModel` at the default 250 ms / 0.5 adverse
fraction, charged on market fills (entry, stop-out, flatten) and not on take
profits, which are limits.

Everything is reported in **R** - multiples of the bracket width, which is the
distance to the stop. That is the only unit in which a $4 bracket from 2023 and
a $27 one from 2026 are the same bet, and the only one that lets gold, FX and an
index share a table. It also makes the results independent of `InpBaseLots` and
of volatility targeting, which change the scale of the equity curve but not its
sign.

**Verdict: do not deploy.** The strategy has no measurable edge over the corpus,
loses significantly over the four-year dev split, and the profitable recent
window is explained by a collapse in cost-per-unit-risk rather than by the
follow-through the header claims. Two of the four presets were selected on data
inside the project's locked test split.

---

## 1. The re-tune was fitted inside the locked split

`qlab.loader.SPLITS` fixes `test` at 2025-07-01 to 2026-09-01, locked, "one
final honest estimate, run once, at the end". The header re-tunes the strategy
*for the 2026 regime* and describes `TOP8_2026` as "validated on only 66
out-of-sample sessions". Both statements place the parameter search inside the
held-out window.

That does not make the strategy wrong, but it does mean the test split can no
longer give this strategy family an honest estimate. The only genuinely
out-of-sample evidence for a 2026-fitted configuration is everything *before*
2026 - which is what the tables below lead with.

## 2. It loses over dev and only works after 2024

`GEO_2026_NO_H13` on XAUUSD, net of all costs:

| period | trades | mean R | total R | daily t | Sharpe | max DD (R) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev 2020-2023 | 6,295 | **-0.057** | -361 | **-2.15** | -1.09 | 450 |
| val 2024-25H1 | 2,604 | +0.081 | +211 | 1.95 | 1.57 | 55 |
| test 25H2-26 | 2,007 | +0.094 | +189 | 2.06 | 1.89 | 45 |
| 2026 (tuned) | 1,140 | +0.147 | +167 | 2.43 | 2.94 | 29 |

The t-statistics are computed on **daily** totals, not per trade: seven windows
on one instrument on one day are mostly the same bet, and a per-trade t-stat
would count that correlation as independent evidence.

The dev result is not a costing artifact. Re-run with dev's own measured cost
profile - which is dearer than the corpus average - it is mean R -0.053,
t = -2.01. A four-year, 6,295-trade losing stretch with a 450R drawdown is the
single most informative fact available about this system, and it sits in the
split the project designated for exactly this purpose.

All four presets show the same shape:

| preset | dev 2020-23 | val 2024-25H1 | test 25H2-26 | 2026 (tuned) |
| --- | ---: | ---: | ---: | ---: |
| GEO_2026_NO_H13 | -0.057 | +0.081 | +0.094 | +0.147 |
| GEO_2026 | -0.045 | +0.079 | +0.088 | +0.132 |
| TOP8_2026 | -0.047 | +0.054 | +0.097 | +0.156 |
| ORIGINAL | -0.068 | +0.107 | +0.054 | +0.053 |

Note the last row. The header's claim that the 2026 geometry "roughly doubles
median P&L against the original" **reproduces in 2026** (+0.147 vs +0.053, a
2.8x improvement) and **reverses in validation** (+0.081 vs +0.107), the one
window that is clean for both configurations. The improvement appears where it
was fitted.

## 3. What actually changed is cost per unit of risk, not follow-through

Decomposing gold by year, per trade:

| year | mean bracket | cost/trade | cost in R | gross R | net R |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020 | $5.51 | $34.70 | 0.089 | +0.094 | +0.005 |
| 2021 | $4.25 | $23.06 | 0.073 | -0.007 | -0.080 |
| 2022 | $4.30 | $21.26 | 0.065 | +0.016 | -0.049 |
| 2023 | $3.97 | $20.14 | 0.068 | -0.024 | -0.091 |
| 2024 | $5.89 | $18.29 | 0.045 | +0.134 | +0.089 |
| 2025 | $12.02 | $16.86 | 0.020 | +0.065 | +0.044 |
| 2026 | $24.03 | $24.38 | **0.013** | +0.160 | +0.147 |

Cost as a share of risk fell **6.7x**, from 0.089R to 0.013R, because the
bracket widened from $5.51 to $24.03 while the dollar cost of a round turn
barely moved. In 2020-2023 the round turn consumed 65-90% of anything the
signal produced; in 2026 it consumes 8%.

This is a real and durable observation, and it is the strongest part of the
header's case - but it is not the case the header actually makes. The stated
rationale is that wider ranges mean *more follow-through*, justifying 60-minute
brackets and 3x targets. That is a different claim, and section 4 tests it
directly. The cost effect, by contrast, is scale-free: it argues that gold in a
high-volatility regime is cheaper to trade at *every* bracket length, and it
favours the **shorter** brackets relatively more, since a fixed cost is a larger
share of a narrower range. It does not select 60 minutes.

For context, the opening range has genuinely widened in relative terms, not just
because gold is dearer - the 60-minute bracket went from 0.28% of price in 2024
to 0.60% in 2026. But the ratio between bracket lengths is unchanged: the
60m/15m range ratio is ~1.9 in every year from 2020 to 2026. The shape of the
range-versus-length curve did not move, only its scale.

## 4. The signal carries no measurable directional information

A backtest's own P&L cannot settle this, because the exit rule is
direction-dependent - winners run to 3R, losers are cut at 1R - so anything
measured from entry to its own exit mixes the signal with the exit's shape.

So: forward return from the moment of entry over a **fixed** horizon, on the
mid, no stop and no target, normalised by the bracket width
(`scripts/research/breakout_attribution.py`). `uncond` is what simply being long over the
same windows returned - the drift a strategy inherits for free.

XAUUSD, 1-hour horizon:

| period | n | uncond | signal | t | edge vs drift | t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-2023 | 6,295 | +0.019 | +0.008 | 0.68 | +0.007 | 0.67 |
| 2024-2026 | 4,611 | +0.021 | +0.024 | 1.82 | +0.022 | 1.71 |
| 2024 | 1,739 | +0.043 | +0.052 | **2.32** | +0.050 | 2.21 |
| 2025 | 1,732 | +0.025 | +0.004 | 0.17 | +0.002 | 0.08 |
| 2026 | 1,140 | -0.016 | +0.011 | 0.43 | +0.012 | 0.47 |

There is no follow-through to speak of in any period, and **none at all in
2026** - the year whose behaviour the re-tune was built to exploit. 2024 is the
only year that clears t = 2, and it does not repeat. Across the other three
instruments the same test gives scattered noise: USDJPY t = 3.10 on dev then
-0.81 in 2026, USTEC t = 2.79 on dev then -1.00 in 2024 and +2.24 in 2026,
EURUSD negative throughout. That is what a set of eight tests on a signal with
no edge looks like.

> **A trap worth recording.** Measuring this forward return from the last
> one-minute bar close before entry, rather than from the true entry mid,
> produces a spurious **+0.13R at t = 11 in every single period**. The bar close
> is up to a minute stale and, because entry happened *by breaking out*, it sits
> on the wrong side of the fill by part of the very move being measured. The
> entry mid is now stored on the trade tape so this cannot recur.

## 5. Hour selection does not survive the regime change

Every UTC hour run at the 2026 geometry (60-minute bracket, 3x target), net
mean R, dev versus 2024-2026:

* Pearson correlation across the 21 tradable hours: **+0.26**
* Spearman correlation of the hour rankings: **+0.36**
* Best-8 hours overlap: **4 of 8** (2.7 expected by chance)

The preset arms hours 0, 1, 2, 4, 5, 6 and 14. In the dev split, hour 5 ranked
**20th of 21** (-0.186R) and hour 1 ranked 12th (-0.076R); in 2024-2026 they
rank 3rd and 1st. The chosen hours are close to the set that happened to work in
the window they were chosen from. With roughly 24 hours x 3 lengths x 3 targets
to search, a t-stat near 2.4 on the tuned window is about what selection alone
produces.

## 6. It does not transfer to any other instrument

Same geometry, same rules, 2020-2026 pooled, net mean R per trade:

| symbol | trades | total R | mean R | dev t | val t | test t | 2026 t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAUUSD | 10,906 | +39 | +0.004 | -2.15 | 1.95 | 2.06 | 2.43 |
| USDJPY | 10,449 | +258 | +0.025 | 1.95 | 0.10 | -0.99 | -2.20 |
| USTEC | 9,523 | -26 | -0.003 | -1.00 | 1.43 | 0.21 | 1.07 |
| EURUSD | 9,622 | -490 | -0.051 | -0.80 | -2.13 | -1.78 | -1.12 |

Nothing is significant in a consistent direction. USDJPY is the mirror image of
gold - positive on dev, negative by 2026. EURUSD loses in every period. If
session opening-range breakout were a structural effect of overnight information
arriving at a session open, some signature would be expected on the index at
least; there is none.

The spread guard was translated as 3.4x each symbol's own mean spread, which is
what `InpMaxSpreadUSD = 0.30` amounts to on gold. Left in dollars it would never
bind on FX. USTEC's daily break (19:00-22:00 UTC) sits under gold's flatten
time, so its late windows are closed rather than expensive.

---

## 7. The MT5 code

The code is better than the strategy. Several things are right that are
routinely wrong:

* **`AdoptExistingState()` is genuinely good.** MT5 destroys and recreates an
  expert on recompile, input edits, timeframe changes and restarts; without this
  pass the expert would stack a second bracket at the same two prices and fill
  both. Rebuilding `hi`/`lo` from the resting order prices, rather than
  persisting them, is the right instinct.
* **The flatten is anchored to the 16:58 New York halt and moves with US DST.**
  The DST arithmetic is correct at both boundaries. A fixed-UTC flatten would
  land inside the summer halt on most days.
* **`CloseStaleFromEarlierSessions()`** is the correct backstop for a
  tick-driven flatten that no tick arrives to trigger.
* **Bar handling is right.** The bracket is `[start, end)` via `to = end - 60`,
  matching MT5's inclusive `CopyRates`, and buy stops trigger on the ask.

Defects, in the order they would cost money:

1. **Position size ignores the stop distance.** The stop *is* the bracket width,
   and the width varies enormously, but `InpBaseLots` is flat. At a fixed 0.02
   lots the dollar risk per trade spans **14x** between its 5th and 95th
   percentiles ($4.78 to $70.31 over 2024-2026). Volatility targeting does not
   fix this - it scales by trailing daily volatility, not by the bracket in
   front of it, so it moves the whole distribution without narrowing it. The fix
   is one line: size inversely to width, `lots ~ risk_budget / (width x
   contract)`. Every R-denominated figure in this report silently assumes it.
2. **No daily loss limit by default.** `InpMaxDailyLossPct = 0.0` is off, on a
   system that can hold eight correlated same-instrument positions.
3. **`InpMaxOpenPositions = 8` with seven or eight windows is not a cap.** The
   "portfolio" is one instrument taking up to eight simultaneous directional
   bets, frequently the same way.
4. **`BuildRange()` failure is permanent for the day.** If `CopyRates` returns
   nothing - normal on a fresh terminal still backfilling M1 history - the
   window sets `armedDay = today` and never retries. Windows silently do not
   trade. A retry with a bounded attempt count would be safer.
5. **`ResolveOCO()` and `CancelWindowPending()` filter orders by magic only, not
   by symbol**, while `CancelAllPending()` and `AdoptExistingState()` filter by
   both. Two instances sharing `InpMagicBase` on different symbols would delete
   each other's orders.
6. **The take profit is re-anchored to the fill; the stop is not.** On a gapped
   entry, realised risk exceeds one bracket width while reward stays 3x from the
   fill. Deliberate, perhaps, but it makes R a nominal rather than a realised
   quantity.
7. `MinutesToUTCMidnight()` is dead code.

**Not a problem, though it looks like one:** the manual OCO is a tick-polled
race - the first fill has to be seen before the opposite stop can be deleted. I
measured it: across 4,611 gold trades in 2024-2026, the opposite bracket edge
was touched within 30 seconds of a fill **zero times**. The other side is a full
bracket width away, so the exposure is theoretical.

---

## 8. What would make this worth another look

The one durable finding here is section 3: cost per unit of risk on gold has
fallen 6.7x, and that is a real structural change in what is affordable to
trade. It just is not evidence for this strategy.

To turn that into something testable:

1. **Size to the stop.** Nothing else can be evaluated until risk per trade is
   constant; at 14x dispersion the equity curve is dominated by which trades
   happened to be large.
2. **Fix the hours before looking at returns.** Pick them from a liquidity or
   information argument - the London and New York opens - and hold them fixed
   across the whole corpus. The current set was chosen by its own P&L.
3. **Test the exit geometry separately from the entry.** The 1R stop / 3R target
   was never independently justified, and section 4 shows the entry signal is
   not what is generating the recent P&L, so the exit is the more likely place
   for anything real to live.
4. **Require a pre-2024 result.** Any configuration that cannot survive
   2020-2023 is a bet on the volatility regime persisting, and should be sized
   and described as one.

The test split should be treated as spent for this strategy family. If a
successor is developed, it needs a new held-out window - the natural choice
being everything after 2026-09-01, forward.

## Reproducing

```bash
python scripts/backtests/backtest_session_breakout.py            # all symbols, all presets
python scripts/research/breakout_attribution.py -s XAUUSD -H 1 4
python -m pytest tests/test_session_breakout.py -q
```

Trade tapes land in `reports/strategies/session_breakout_trades/`, one parquet
per symbol and preset, with per-trade entry and exit mids so the attribution can
be recomputed without a re-run.
