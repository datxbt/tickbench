# The engulfing bar's quadrants - evaluation

Subject: the engulfing bar traded from its own fibonacci quadrants — sell the
retracement into the 0.25–0.5 zone, target the bar's low.

* **Condition 1** — `high[0] > high[1]`, the bar takes out the prior candle's high
* **Condition 2** — `close[0] < open[1]`, and closes below the prior candle's open
* **Quadrants** — the signal bar divided into four equal quarters by `0, 0.25,
  0.5, 0.75, 1`, anchored with **0 at the bar's low and 1 at its high**
* **Entry** — a resting order worked in the `0.25–0.5` zone
* **Target** — `0`, the bar's low
* **Stop** — not specified by the idea, so swept: `0.75`, `1` (the bar's high)
  and `1.25`
* **Size** — 0.01 lot, one position at a time

This is **not** the body-engulfing candle rejected in
[engulfing.md](engulfing.md). Condition 1 is about the *wick*, so the bar has to
physically trade through the previous high before turning, and condition 2
references the previous *open* rather than its body's far edge. The two together
describe a stop run — highs taken, then given back past where the prior bar
started. It gets its own module, its own tape walk and its own null.

Method: implemented in `src/qlab/strategies/engulfing_quadrant.py` and run
tick-by-tick over 2020-01 to 2025-06 on all four instruments at **eleven
timeframes from 1m to 4h** — 1m, 2m, 3m, 5m, 10m, 15m, 20m, 30m, 1h, 2h, 4h,
the same ladder the body-engulfing study swept so the two sit on one axis.
**1,386,322 signals, 784,381 fills, 772,796 trades at the stated geometry.**
Fills cross a real bid and a real ask from the tape; commission is the published
contract term; slippage is `qlab.costs.CostModel` at the default 250 ms / 0.5
adverse fraction, charged on every market leg and never on a resting limit,
which fills at its price or not at all.

The idea nominates one entry, one target and no stop, so none of the three is
privileged: **three entry conventions × three stops × five targets = 45 exit
cells per instrument-timeframe**, all resolved against the same tape in one pass.

> **Verdict: do not trade this. Rejected on dev, rejected again on validation.
> The held-out test split was not spent.**
>
> Pooled over both splits the setup loses **0.502R per trade** and **−$76,940**
> at 0.01 lot, is positive in **1 of 88** (instrument × timeframe × split) cells,
> and **75 of 88** sit at t ≤ −2. Of 1,320 (cell × stop × target) combinations,
> 60 are positive and **none clears t = +2**.
>
> The rejection is not empirical. Section 1 shows that for **every** entry, stop
> and target this geometry can take, the break-even strike rate and the
> fair-coin strike rate are **exactly equal at zero cost** — the trade is a fair
> bet by construction, and the round turn can only push it below zero. The
> eleven-rung ladder in section 2 is that algebra playing out: as cost/R falls
> from 0.52 to 0.03, the realised strike rate closes on the coin and never on
> break-even.

---

## 1. The geometry is a fair bet by construction

Entry between 0.25 and 0.5 with the target at 0 means the reward is **a quarter
to a half of one bar range**. Against a stop at the bar's high the risk is 0.5
to 0.75 of that range, so the trade is **0.33R to 1.0R** and needs a 50% to 75%
strike rate before a single cost is charged. In practice the entry lands at a
mean fib of **0.334**, implying a payoff of **0.501R** and a break-even rate of
**66.6%**.

That is not a flaw. It is the design, and it would be fine if the pattern beat
it. The reason it cannot is an identity.

Write `e` for the entry level, `s` for the stop and `t` for the target, all on
the bar's own 0–1 ladder. Then

```
payoff            = (e − t) / (s − e)
break-even rate   = (1 + cost) / (1 + payoff)
fair-coin rate    = (s − e) / (s − t)      # driftless first passage
```

and at zero cost the second collapses onto the third:

```
1 / (1 + (e−t)/(s−e))  =  (s−e) / ((s−e) + (e−t))  =  (s−e) / (s−t)
```

**For every (entry, stop, target) triple, the strike rate needed to break even
with no costs is exactly the strike rate a driftless random walk delivers.**
Verified in exact rational arithmetic across the whole swept grid — maximum
difference 0 — and pinned by `test_breakeven_equals_fair_coin_at_zero_cost`.

The consequence is the whole study. A barrier pair placed on a driftless walk is
a fair bet whatever levels you choose, so *no* entry zone, stop, target or
timeframe can create an edge here; only a pattern that makes the walk non-driftless
could, and section 6 shows this one does not. Costs then make a fair bet a losing
one. That is why the 15-cell exit grid is flat, why every rung of the ladder
loses, and why this setup cannot be rescued by tuning.

## 2. The timeframe ladder

The sweep, pooled over the four instruments, at the stated geometry. `needs` is
the break-even strike rate; `fair` is what a coin pays on the same barriers;
`vs BE` and `vs fair` are the realised rate minus each.

**Dev (2020-01 to 2023-12)**

| tf | signals | trades | fill% | miss% | payoff | cost/R | needs | fair | actual | vs BE | vs fair | mean R | t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1m | 403,805 | 209,016 | 0.525 | 0.379 | 0.413 | 0.520 | **1.076** | 0.670 | 0.526 | −0.550 | −0.144 | −0.727 | −65.4 |
| 2m | 217,295 | 118,142 | 0.551 | 0.361 | 0.428 | 0.370 | 0.959 | 0.667 | 0.559 | −0.400 | −0.108 | −0.533 | −57.2 |
| 3m | 147,869 | 82,186 | 0.564 | 0.351 | 0.444 | 0.302 | 0.902 | 0.664 | 0.573 | −0.328 | −0.090 | −0.444 | −50.1 |
| 5m | 90,769 | 52,485 | 0.587 | 0.335 | 0.455 | 0.233 | 0.847 | 0.664 | 0.590 | −0.258 | −0.074 | −0.356 | −39.6 |
| 10m | 46,501 | 27,815 | 0.608 | 0.321 | 0.469 | 0.167 | 0.794 | 0.662 | 0.602 | −0.193 | −0.061 | −0.271 | −26.5 |
| 15m | 31,331 | 19,164 | 0.621 | 0.315 | 0.476 | 0.135 | 0.769 | 0.661 | 0.614 | −0.155 | −0.047 | −0.219 | −18.3 |
| 20m | 23,562 | 14,453 | 0.624 | 0.314 | 0.483 | 0.119 | 0.754 | 0.661 | 0.616 | −0.139 | −0.045 | −0.199 | −14.9 |
| 30m | 15,618 | 9,951 | 0.650 | 0.292 | 0.487 | 0.095 | 0.737 | 0.660 | 0.615 | −0.122 | −0.045 | −0.176 | −11.0 |
| 1h | 7,993 | 5,171 | 0.662 | 0.282 | 0.494 | 0.065 | 0.713 | 0.660 | 0.637 | −0.075 | −0.022 | −0.107 | −5.2 |
| 2h | 4,521 | 2,936 | 0.672 | 0.279 | 0.492 | 0.042 | 0.698 | 0.660 | 0.654 | −0.044 | −0.006 | −0.065 | −2.4 |
| 4h | 2,422 | 1,635 | 0.697 | 0.262 | 0.500 | 0.028 | 0.686 | 0.660 | 0.645 | −0.040 | −0.015 | −0.060 | −1.6 |

**Validation (2024-01 to 2025-06)**

| tf | signals | trades | fill% | miss% | payoff | cost/R | needs | fair | actual | vs BE | vs fair | mean R | t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1m | 165,031 | 92,811 | 0.571 | 0.335 | 0.404 | 0.417 | **1.010** | 0.665 | 0.582 | −0.428 | −0.083 | −0.580 | −34.7 |
| 2m | 86,090 | 50,311 | 0.593 | 0.327 | 0.437 | 0.304 | 0.908 | 0.662 | 0.596 | −0.311 | −0.065 | −0.435 | −29.6 |
| 3m | 58,085 | 34,241 | 0.598 | 0.326 | 0.444 | 0.252 | 0.867 | 0.662 | 0.604 | −0.263 | −0.058 | −0.369 | −27.1 |
| 5m | 35,146 | 21,107 | 0.609 | 0.318 | 0.458 | 0.202 | 0.825 | 0.660 | 0.615 | −0.210 | −0.045 | −0.301 | −20.4 |
| 10m | 17,946 | 11,020 | 0.624 | 0.310 | 0.473 | 0.145 | 0.777 | 0.659 | 0.620 | −0.157 | −0.039 | −0.230 | −13.6 |
| 15m | 11,818 | 7,416 | 0.640 | 0.299 | 0.476 | 0.116 | 0.756 | 0.659 | 0.622 | −0.134 | −0.037 | −0.201 | −10.3 |
| 20m | 8,897 | 5,554 | 0.637 | 0.301 | 0.487 | 0.101 | 0.741 | 0.657 | 0.624 | −0.116 | −0.033 | −0.174 | −8.2 |
| 30m | 5,985 | 3,795 | 0.649 | 0.296 | 0.481 | 0.083 | 0.731 | 0.659 | 0.635 | −0.096 | −0.024 | −0.143 | −5.7 |
| 1h | 3,080 | 1,967 | 0.652 | 0.294 | 0.496 | 0.057 | 0.707 | 0.658 | 0.664 | −0.043 | **+0.006** | −0.064 | −1.9 |
| 2h | 1,664 | 1,057 | 0.653 | 0.284 | 0.494 | 0.035 | 0.693 | 0.663 | 0.635 | −0.058 | −0.029 | −0.088 | −1.9 |
| 4h | 894 | 563 | 0.657 | 0.286 | 0.510 | 0.025 | 0.679 | 0.653 | 0.659 | −0.020 | **+0.006** | −0.031 | −0.5 |

Every column is monotone in the timeframe, and the two comparison columns say
different things:

* **`vs fair` closes on zero** — from −0.144 to −0.006 on dev, −0.083 to +0.006
  on validation. As the round turn shrinks the setup converges on the coin, from
  below, exactly as section 1 predicts.
* **`vs BE` never closes** — −0.550 to −0.040 on dev, −0.428 to −0.020 on
  validation. It cannot: `needs` sits above `fair` by roughly `cost/R × fair`,
  and the gap only reaches zero where the round turn does.

The two fastest rungs deserve their own line. At 1m the break-even rate is
**above 1.0** on both splits pooled (1.076 dev, 1.010 validation), and at 2m it
is 0.959 and 0.908. A rate above 1 means the round turn exceeds the entire
payoff and **the trade cannot be won at any strike rate, including a perfect
one.** No amount of pattern quality is relevant below about 3m.

At the slow end the loss becomes statistically indistinguishable from zero
(4h: t = −1.6 dev, −0.5 validation) — but it is indistinguishable from zero
*because it has converged on a fair coin*, not because an edge has appeared.

## 3. The resting order misses the winners

The order sits in a zone the bar has already left, and the bar closes low — the
median close sits at 0.10 to 0.27 of its own range. So it usually rests *above*
the market as a sell limit, waiting for a bounce that may not come.

Dev, at the stated zone entry, over all 991,686 signals:

| outcome | share | what it is |
| --- | ---: | --- |
| **filled** | 55.6% | the order was worked |
| **missed** | 35.8% | price reached the bar's low without ever returning to the zone |
| **invalidated** | 8.6% | price traded back above the bar's high first |
| **expired** | 0.02% | neither, within 12 bars |

**More than a third of signals are missed, and they are the ones the idea would
have won.** Their median close sits at fib 0.033 — those bars closed essentially
*on* their low, the strongest-looking version of the pattern, and precisely the
case where price never comes back to let you in. The invalidated signals are the
opposite: median close at fib 0.500, at the top of the zone, filling easily and
then running over the stop.

The miss rate falls monotonically with the timeframe (37.9% at 1m to 26.2% at
4h) but never becomes small. The entry convention selects *against* the trader
at every rung, and a backtest that tests the target before the entry books all
354,858 dev misses as winners.

The 12-bar order life is not what is doing this: **224 signals out of 991,686
expired**, so the deadline essentially never binds.

## 4. What one unit of risk costs

Risk here is the entry level to the stop level, so 1R is a *fraction* of one bar
range — about 0.67 of it at the stated stop — where the body-engulfing study's
1R was a whole one. Round-turn cost as a fraction of the amount risked, median
over dev:

| | 1m | 3m | 5m | 15m | 30m | 1h | 4h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 0.511 | 0.305 | 0.237 | 0.140 | 0.099 | 0.069 | 0.029 |
| USDJPY | 0.603 | 0.360 | 0.281 | 0.162 | 0.116 | 0.074 | 0.033 |
| XAUUSD | 0.530 | 0.298 | 0.228 | 0.128 | 0.090 | 0.061 | 0.028 |
| USTEC | 0.451 | 0.258 | 0.191 | 0.108 | 0.074 | 0.053 | 0.023 |

The resting limit does buy something real: it does not cross the spread on
entry, which a market order does. But the risk it is measured against is
two-thirds of a bar range rather than a whole one, and that more than cancels
the saving. On one-minute bars the median trade pays **45% to 60% of everything
it risks** to get in and out.

## 5. The pattern predicts nothing at the horizon the trade lives on

Strip the order and the barriers off and measure the unconditional forward move
from the bar's close, signed short. No entry zone, no stop, no target, no cost:

* The 5-bar forward return is **negative in 35 of 44 dev cells**, median
  −0.095 bps — the right sign for the setup, and far too small to matter against
  a round turn of 0.5R.
* Maximum favourable and maximum adverse excursion over the next 20 bars, in
  bar-range units, are **the same size in every single cell**. Price runs as far
  against the signal as it runs for it.

There *is* a real drift at the 20-bar horizon: negative in 42 of 44 dev cells,
median −0.400 bps, reaching −26.9 bps at t = −2.10 on USTEC 4h. That is a
genuine slow move in the direction the setup takes — but it is a *20-bar* move,
and the setup's target sits a third of one bar away and resolves within a bar or
two. **The strategy cannot reach the only thing the signal predicts.** Capturing
it would need a different exit entirely, and would be a slow-timeframe drift
study rather than this one.

## 6. The nulls, and the trap in comparing them

Two nulls, both keeping the entire quadrant machinery and dropping only the
pattern. `matched` draws a sample of ordinary bars. `nosweep` is the sharper
one: bars that close below the prior open but **did not** take out the prior
high — condition 2 without condition 1, which isolates what the stop run itself
is worth. Both are drawn size-matched to the signal by a hash of the bar's own
timestamp, so they are deterministic across runs and carry the same weight in
every pooled figure.

On raw mean R the setup beats both, in **160 of 176** cell comparisons:

| pooled dev + validation, stated geometry | signal | matched | nosweep |
| --- | ---: | ---: | ---: |
| mean R | **−0.502** | −0.681 | −0.640 |

**That comparison is invalid, and documenting it is the point of this study.**
The variants do not share a geometry. Signal bars close near their lows, so the
entry lands lower in the bar than a control bar's does — which *simultaneously*
lowers the payoff and raises the fair-coin strike rate. The setup is being
credited for standing closer to its target, not for predicting anything. By
section 1 all three are fair bets; only their levels differ.

Score each against **its own** fair-coin value and the advantage disappears:

| pooled dev + validation | trades | win | fair | **win − fair** |
| --- | ---: | ---: | ---: | ---: |
| **signal** | 772,796 | 0.571 | 0.665 | **−0.094** |
| matched | 737,682 | 0.519 | 0.596 | −0.076 |
| nosweep | 784,774 | 0.543 | 0.632 | −0.089 |

The signal is the **worst of the three** on the pooled trade-weighted number,
and on both splits taken separately. Cell by cell it beats the matched null on
`win − fair` in only 26 of 88 and the no-sweep null in 34 of 88 — a coin flip at
best. The 160-of-176 raw advantage does not survive the normalisation.

The `nosweep` comparison is the one that speaks to the pattern's own logic. Its
bars close below the prior open *without* having run the prior high, so they are
the setup minus the stop run and nothing else. They land 0.089 below their fair
coin; the setup lands 0.094 below its own. **The stop run is worth −0.005, which
is to say nothing.** What was being measured is where the close sits in the bar,
which every candle in the sample has.

## 7. The mirror reproduces it exactly, and the stricter reading is worse

A pattern that predicts direction must behave differently when reflected. The
mirror — takes out the prior *low*, closes *above* the prior open, entry in the
0.25–0.5 zone measured down from the high, target the high — is the same
geometry pointed the other way.

| pooled dev + validation | trades | mean R | win | fair | win − fair |
| --- | ---: | ---: | ---: | ---: | ---: |
| signal (bearish) | 772,796 | −0.5019 | 0.571 | 0.665 | −0.0944 |
| **mirror (bullish)** | 774,303 | **−0.5012** | 0.573 | 0.667 | **−0.0942** |
| fade (placebo, same bars) | 609,658 | −0.6444 | 0.475 | 0.535 | −0.0600 |
| prior candle bullish | 313,662 | −0.4352 | 0.597 | 0.708 | −0.1106 |

The bearish setup and its bullish mirror agree to **0.0007R** and 0.0002 on the
normalised edge, over 2020–2025 — a period containing a gold bull market and a
USTEC drawdown. A directional edge could not be this symmetric.

The last row is the same illusion as section 6 in miniature. The rules never say
the prior candle must be bullish, and it is one only about half the time.
Requiring it gives a **higher** raw strike rate (0.597 vs 0.571) and a better
raw mean R (−0.435 vs −0.502), which looks like the filter helping — but its
entry lands lower still, so its fair-coin rate rises to 0.708 and its normalised
edge is **−0.111**, the worst of every variant tested. Making the pattern
stricter makes it worse.

## 8. No exit rescues it

The grid is 44 instrument-timeframe cells × 15 (stop, target) pairs × 2 splits =
1,320 combinations. **60 are positive; none clears t = +2.**

Mean R at the stated stop, averaged over the 44 dev cells, as the target moves
from the bar's low down to a full range below it:

| target | 0 (stated) | −0.25 | −0.5 | −1 | hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean R | −0.288 | −0.280 | −0.278 | −0.275 | −0.237 |

The curve is **flat** — moving the target by a whole bar range changes the mean
by 0.014R while the cost alone is 0.03 to 0.60R. Section 1 says why: every one
of these placements is a fair bet before costs, so they differ only in how much
cost each pays. The mild improvement toward `hold` is not an edge either; it is
fewer round turns per unit of time.

## 9. The three readings of "enter in the zone"

"Enter in the zone" is genuinely ambiguous, so all three readings were priced.
`zone` fills at whichever edge price reaches first; `q25` and `q50` rest at a
fixed level. Dev, at the stated stop and target:

| | fill rate | mean R | win | fair | win − fair |
| --- | ---: | ---: | ---: | ---: | ---: |
| zone (first touch) | 0.556 | −0.529 | 0.560 | 0.666 | −0.107 |
| q25 (shallow) | 0.503 | −0.506 | 0.618 | 0.750 | −0.132 |
| q50 (deep) | 0.392 | −0.728 | 0.385 | 0.500 | −0.115 |

Note how the three fair-coin values differ — 0.75, 0.67, 0.50 — exactly tracking
the entry level, and how all three realised rates sit well below their
own — 10.7, 13.2 and 11.5 points. That is the same answer three times, which
is what section 1 guarantees.
Waiting for the better price is worse in raw terms: `q50` sells half a bar range
higher — a 1.0R payoff against 0.33R — and still loses most, because requiring a
deeper retracement drops the fill rate to 39% and selects for bars whose bounce
had momentum behind it.

Of the 132 (cell × entry mode) combinations at the stated geometry, **one is
positive on dev** — EURUSD 4h at `q50`, +0.010R over 278 trades, t = +0.17 — and
four on validation, all at 4h, the best at t = +0.62. Nothing was legitimately
selected on dev, so nothing earned a validation confirmation, let alone the test
split.

## 10. Validation

Validation is milder and says the same thing. Spreads have compressed since
2020, so `cost/R` falls and every number moves toward zero without changing sign:

| stated geometry | dev | validation | pooled |
| --- | ---: | ---: | ---: |
| trades | 542,954 | 229,842 | 772,796 |
| mean R | −0.529 | −0.437 | −0.502 |
| strike rate | 0.560 | 0.598 | 0.571 |
| fair-coin rate | 0.666 | 0.663 | 0.665 |
| **win − fair** | **−0.107** | **−0.065** | **−0.094** |
| below fair-coin | 41/44 | 37/44 | 78/88 |
| below break-even | 44/44 | 43/44 | 87/88 |
| cells positive | 0/44 | 1/44 | 1/88 |
| cells at t ≤ −2 | 39/44 | 36/44 | 75/88 |

Judged on the pooled splits rather than the friendlier one, the setup is
negative on every axis and beaten by its null.

## 11. What would have to be true

1. **The pattern makes the walk non-driftless at the horizon the trade lives
   on.** It does not: the 5-bar forward return is −0.095 bps against a 0.5R round
   turn, excursions are symmetric, and the bullish mirror reproduces the bearish
   numbers to 0.0007R. This is the *only* route to an edge here, because of the
   identity in section 1.
2. **The retracement entry gets a better price than the risk it adds.** It does
   not: 36% of signals never come back, and they are disproportionately the ones
   that would have won.
3. **Some (stop, target) pair extracts value the signal has.** There is no value
   to extract, and section 1 proves no pair can: all 1,320 combinations are fair
   bets before costs, and none of them clears t = +2 after.
4. **A slow timeframe escapes the cost.** The ladder shows 4h escaping the cost
   — and landing on the fair-coin strike rate, which is the same as having no
   edge.

## 12. Reproducing

```bash
python scripts/backtests/backtest_engulfing_quadrant.py                       # 1m -> 4h, all four
python scripts/backtests/backtest_engulfing_quadrant.py --controls            # both nulls
python scripts/backtests/backtest_engulfing_quadrant.py --controls matched    # skip the 6x-size null
python scripts/backtests/backtest_engulfing_quadrant.py --variants            # fade, mirror, prior-bull
python scripts/backtests/backtest_engulfing_quadrant.py --from-tapes          # re-summarise, no tape walk
```

Artifacts: `reports/strategies/engulfing_quadrant.json` (every table above as
data) and `reports/strategies/engulfing_quadrant_trades/*.parquet` (the full
trade tapes, one row per signal per entry mode per variant per split). Tests in
`tests/test_engulfing_quadrant.py` pin the pattern's two conditions and the
near-misses that must not fire, the fibonacci anchor on both sides of the
market, the break-even/fair-coin identity of section 1, which side of the book
each level acts on, that a limit fills at its price while a stop pays slippage,
and — the one that matters most — that a signal whose price reached the target
without ever returning to the zone is recorded as `missed` rather than as a win.

**The test split (2025-07-01 onward) remains unspent.**
