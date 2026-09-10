# Structural-break entries on gold - component decomposition

Subject: Quan, Zhong and Jiang's H4 XAUUSD swing-break system, and the three
separable claims inside it.

1. A confirmed pivot in the H4 closes marks a **structural break**, and a stop
   order resting at it carries alpha. Stop 3 dollars, target 20 dollars.
2. **Regime-conditional sizing** (V11) adds more than leverage does.
3. A **half-exit at +1R on a counter-directional minute**, with the stop then at
   breakeven, improves the risk-adjusted return.

Method: implemented in `src/qlab/strategies/structural_break.py` as one fixed
core with four independent axes - entry, sizing, exit, and the resolution of the
backtest itself - moved one at a time. Run over the dev split's **6,262 H4 bars
and 100.2 million real ticks**, then once on validation. Fills cross a real bid
and a real ask; commission is the published contract term; slippage is
`qlab.costs.CostModel`, charged on market fills and never on a take profit.
Every headline comparison is a **paired** stationary-block bootstrap on the
common daily return grid.

**Verdict: claim 3 is not merely unsupported, it is backwards - the half-exit is
the single most destructive component in the system, and removing it is worth
about one full unit of Sharpe (ΔSharpe -1.07, 95% CI [-1.65, -0.48], p < 0.0001,
survives Bonferroni). Claim 2 is leverage: no sizing rule moves Sharpe
significantly and the 2x variants confirm the invariance exactly. Claim 1 is
unproven: against a 200-path random-entry null the break sits at the 51st
percentile with the overlay on and the 78th with it off (one-sided p = 0.49 and
0.23), and with the overlay removed it is break-even on dev (mean R +0.004) and
positive on validation (+30.5%, Sharpe 0.67) - which is not a survivor of dev
and did not earn the test split.** The candle filter - the
sub-component the specification isolates as `V11_FixedR` - carries nothing at
all (ΔSharpe +0.024, p = 0.94).

---

## 0. The one genuinely ambiguous instruction, and why both readings are run

Step 3 of the entry says: *if the H4 close pierces the most recent confirmed
swing high, place a pending buy-stop order at that broken level.* Taken
literally, the order is placed **after** price is already above the level, where
a buy stop cannot rest - it needs a pullback and a re-break first.

The other reading is that the order rests at the pivot from the moment the pivot
is confirmed, and the "close pierces" clause is the bar-resolution description
of it filling. That reading is always placeable: a swing high is by construction
a close no later close in its confirmation window exceeds, so at the confirming
bar the price is at or below it.

Both are implemented (`arm="on_confirm"`, `arm="retest"`) and both are reported.
They are very different strategies:

| arming | trades on dev | mean R | orders expired unfilled |
| --- | ---: | ---: | ---: |
| `on_confirm` | 2,048 | -0.073 | 0 |
| `retest` | 258 | -0.114 | 287 |

Neither works, so nothing below turns on the choice, and `on_confirm` is used as
the default because it is the one that corresponds to an order actually sitting
in a book.

Two other places where the specification left a choice and one had to be made,
both stated rather than tuned:

- **V11's multipliers** are described only as "calibrated lookups". A
  calibration cannot be transcribed, so `m_ATR = 1/rho` (shrink when
  short-horizon volatility runs hot relative to its own baseline) and a
  three-state trend multiplier (1.25 aligned / 0.75 opposed / 1.00 flat) were
  chosen. Section 3 shows why a different calibration would not change the
  conclusion.
- **The half-exit fires at a minute close.** Its trigger reads "the current M1
  candle is counter-directional", and a candle has no direction until it closes.
  Evaluating it at the close is not a simplification, it is the only
  non-anticipating reading - and it is why the stop and the target, which are
  resting orders, take priority inside the minute they share with it.

---

## 1. Axis 2 first, because it is the largest effect in the study

Dev split, real ticks, V10 sizing, entry held fixed:

| run | n | return% | Sharpe | Sortino | Calmar | maxDD% | win% | payoff | PF | mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V10 half-exit on | 2,048 | -43.3 | -1.08 | -1.80 | -0.28 | -46.8 | 41.2 | 1.24 | 0.87 | -0.073 |
| **B5 half-exit off** | 1,666 | **-5.3** | **-0.01** | -0.02 | -0.04 | -34.5 | 13.8 | **6.18** | 0.99 | **+0.004** |
| V11_FixedR no filter | 2,081 | -42.8 | -1.11 | -1.76 | -0.28 | -46.4 | 44.1 | 1.10 | 0.87 | -0.071 |

Paired bootstrap on the common daily grid, 1,000 resamples:

| comparison | statistic | point | 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| V10 vs B5 | ΔSharpe | **-1.070** | [-1.652, -0.481] | **0.0000** |
| V10 vs B5 | ΔCalmar | -0.211 | [-0.931, -0.022] | 0.0080 |
| V10 vs B5 | ΔMaxDD | -0.123 | [-0.252, +0.057] | 0.2980 |
| V10 vs FixedR | ΔSharpe | +0.024 | [-0.367, +0.389] | 0.9960 |
| V10 vs FixedR | ΔCalmar | -0.007 | [-0.066, +0.065] | 0.8160 |
| V10 vs FixedR | ΔMaxDD | -0.004 | [-0.074, +0.057] | 0.6720 |

Family B corrected together: the ΔSharpe and ΔCalmar results against B5 survive
Bonferroni (0.0000 and 0.0480); everything involving the candle filter is above
0.67 under any correction.

**The mechanism is arithmetic, and the win rate is what gives it away.** The
overlay lifts the win rate from 13.8% to 41.2% - a 27.4 percentage-point gain,
z = +18.29, p < 0.0001 - and collapses the payoff ratio from 6.18 to 1.24. It
fires on **99.9% of winners and 0.1% of losers**. That is not a filter, it is a
tautology: a rule that triggers at +1R can only ever fire on a trade that
reached +1R, and on this geometry a trade that reached +1R was most of the way
to being a winner.

What it does to such a trade is cap it. A full 20-dollar target on a 3-dollar
stop is 6.67R; take half at +1R and the same trade pays roughly
`0.5 x 1 + 0.5 x 6.67 = 3.8R`. The system's entire expectancy lives in a small
number of large winners - 13.8% win rate at a 6.18 payoff - and the overlay
halves exactly those. It makes the equity curve feel better every single day and
destroys the only property that made the system viable.

This is the disposition effect the specification says the overlay "encodes". It
encodes it faithfully. That is the problem.

**The candle filter does nothing whatsoever.** Comparing V10 to V11_FixedR
isolates the counter-directional-candle condition from the +1R threshold, and
the two are indistinguishable on every statistic. The condition changes which
minute the half fires on and changes no outcome. Section 4 shows this conclusion
is not an artifact of backtest resolution either, which was the obvious
objection to it.

## 2. Axis 4: the backtest's resolution is worth more than most of the components

Three exit conditions - stop, target, and the sub-bar overlay - can come due
inside the same minute, and OHLC cannot say which came first. Real ticks can.

| cell | mode | n | return% | Sharpe | mean R | Δ mean R vs ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V10 half-exit on | ticks | 2,048 | -43.3 | -1.08 | -0.073 | - |
| | interp | 2,055 | -39.7 | -1.08 | -0.065 | +0.0084 |
| | close | 1,885 | -41.9 | -0.97 | -0.075 | -0.0025 |
| **B5 half-exit off** | **ticks** | 1,666 | **-5.3** | **-0.01** | **+0.004** | **-** |
| | interp | 1,690 | -24.1 | -0.36 | -0.034 | **-0.0377** |
| | close | 1,510 | -24.8 | -0.25 | -0.039 | -0.0426 |
| V11_FixedR | ticks | 2,081 | -42.8 | -1.11 | -0.071 | - |
| | interp | 2,081 | -42.7 | -1.21 | -0.071 | +0.0002 |
| | close | 1,931 | -40.3 | -0.95 | -0.070 | +0.0009 |

The B5 row is the finding, and it runs the *opposite* way from the usual warning
about bar-level backtests. A minute-bar tape traversed in candle-direction order
(`open -> low -> high -> close` on an up minute) systematically resolves the
stop before the target on a long, because the low is visited first. On a
geometry with a 3-dollar stop and a 20-dollar target that assumption bites
constantly, and it turns a break-even system into a clearly losing one: -0.038 R
a trade, which on 1,666 trades is the whole result.

So a bar-level backtest of this strategy **understates** it by more than the
strategy is worth. The specification's own framing - that interpolated ticks
overstate performance - does not hold for this geometry; the direction of the
error depends on which side of the position the near barrier is on, and here it
is the stop.

The difference-in-differences that matters most is whether the candle-filter
result is itself a resolution artifact. It is not:

| mode | ΔSharpe (V10 - FixedR) | 95% CI | p |
| --- | ---: | ---: | ---: |
| ticks | +0.024 | [-0.345, +0.424] | 0.938 |
| interp | +0.130 | [-0.171, +0.453] | 0.378 |

Zero on the honest tape, and still zero on the coarse one.

## 3. Axis 3: the sizing rules are leverage

Dev, ticks, V10 exits and entry held fixed. Both risk levels:

| rule | 1x return% | 1x Sharpe | 2x return% | 2x Sharpe | 2x/1x return |
| --- | ---: | ---: | ---: | ---: | ---: |
| V10 | -43.3 | -1.084 | -69.7 | -1.091 | 1.64 |
| V11 | -39.7 | -0.882 | -66.1 | -0.890 | 1.70 |
| V11_ATR | -39.4 | -0.918 | -65.3 | -0.919 | 1.69 |
| V11_TREND | -44.0 | -1.059 | -70.6 | -1.064 | 1.63 |
| V11_VOLTGT | -46.6 | -0.993 | -73.7 | -0.991 | 1.61 |

The invariance check the specification asks for is wired into the harness rather
than hidden in a test file, and it holds to the last bit: `Sharpe(r)` and
`Sharpe(3.7 r)` differ by 0.00e+00. So the 2x columns are the control. Doubling
the risk moves the return by 1.6x (compounding on a losing curve, not 2x) and
moves Sharpe by less than 0.01 in every row. That is what pure leverage looks
like.

Against that control, Family C:

| rule vs V10 | ΔSharpe | 95% CI | p | Bonferroni |
| --- | ---: | ---: | ---: | ---: |
| V11 | +0.202 | [-0.125, +0.538] | 0.2360 | 0.9440 |
| V11_ATR | +0.166 | [-0.038, +0.364] | 0.1020 | 0.4080 |
| V11_TREND | +0.025 | [-0.286, +0.301] | 0.8960 | 1.0000 |
| V11_VOLTGT | +0.091 | [-0.256, +0.441] | 0.6200 | 1.0000 |

Every interval contains zero. The ATR term is the only one with a hint of
anything (+0.166, and its interval is the narrowest of the four), and it is what
you would expect: sizing down when short-horizon volatility is elevated is a
mild risk improvement, not an alpha. The trend term - the half of V11 that
actually expresses a directional view - contributes +0.025, which is nothing.

Two caveats worth stating plainly. These deltas are measured on a **losing**
base strategy, where a Sharpe improvement can come from losing less rather than
from earning more; and the multipliers are a calibration this study chose. What
neither caveat touches is the invariance: a rule can only add something by
changing the *pattern* of exposure relative to subsequent realised volatility,
and no rule here does that at a magnitude the data can see.

## 4. Axis 1: the entry

Every row here has the identical stop, target, sizing rule and exit overlay.
Only the instant and the side of the entry differ, so a difference between rows
is a statement about the entry and nothing else.

The axis is run twice, because with the overlay on it cannot say much: sections
1 and 2 show the overlay dominates the arithmetic, so both the strategy and its
null are measured through it. The comparison is still fair - the null gets the
same overlay - but the level is uninformative. With the overlay off, the entry
is the only thing left.

**With the overlay on (V10), dev, real ticks:**

| run | n | return% | Sharpe | win% | payoff | mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| break (on_confirm) | 2,048 | -43.3 | -1.08 | 41.2 | 1.24 | -0.073 |
| MA(20,50) crossover | 134 | -4.4 | -0.27 | 38.8 | 1.34 | -0.090 |
| break (retest) | 258 | -10.3 | -0.66 | 43.8 | 1.02 | -0.114 |

**With the overlay off (B5), dev, real ticks:**

| run | n | return% | Sharpe | win% | payoff | mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **break (on_confirm)** | 1,666 | **-5.3** | **-0.01** | 13.8 | 6.18 | **+0.004** |
| MA(20,50) crossover | 131 | -5.8 | -0.18 | 12.2 | 6.18 | -0.116 |
| break (retest) | 232 | -17.1 | -0.67 | 10.8 | 6.28 | -0.215 |

The structural break is the best of the three entries once the overlay is out of
the way - it beats a naive moving-average crossover by 0.12 R a trade and its own
literal-reading `retest` variant by 0.22 R. That is the only thing in this study
that points the specification's way.

It is not, however, better than doing nothing. **Buy and hold gold over the same
window returns +30.8% at Sharpe +0.51** (t = +1.03), against the strategy's
-5.3%. An entry whose best case is break-even, on an instrument that rose 31%
underneath it, has not demonstrated that it is finding structure.

**The random-entry null (B.5, 200 paths).** Same trade count, same clock, same
stop and target, random timing and random side. Run on the interpolated tape,
which understates both sides equally (section 2), because a thousand tick-mode
paths would take the better part of an hour.

With the V10 overlay the strategy sits at the **51st percentile** of its own
null - mean R -0.0646 against a null mean of -0.0668, sd 0.0371, and a one-sided
p of 0.49. Dead centre. Corrected together as Family A, no comparison comes
close: versus the MA crossover ΔSharpe -0.82 (p = 0.23), versus the retest arm
-0.42 (p = 0.55), versus random entry p = 0.49; Bonferroni 0.68, 1.00, 1.00.

With the overlay off (`--entry-exits b5`) the entry finally separates from its
null, and not by enough:

| null, 200 paths, B5 exits | strategy | null mean | null sd | null p05 | null p95 | percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean R / trade | -0.0336 | -0.0851 | 0.0622 | -0.1936 | +0.0099 | **0.775** |
| total return | -0.2412 | -0.3133 | 0.1767 | -0.5866 | -0.0103 | 0.660 |

The 78th percentile, one-sided p = 0.225. That is the right direction and it is
not a result. Note that both figures are measured on the interpolated tape,
where section 2 showed B5's mean R is understated by 0.038 - the strategy and
its null are measured the same way, so the *comparison* is sound, but the
levels in this table are not the tick-mode levels and should not be read across
to section 1.

Family A, all three entry comparisons corrected together (B5 exits):

| comparison | ΔSharpe | p | Bonferroni | BH |
| --- | ---: | ---: | ---: | ---: |
| vs MA crossover | +0.17 | 0.7040 | 1.0000 | 0.7040 |
| vs break retest | +0.65 | 0.3080 | 0.9240 | 0.4620 |
| vs random entry | - | 0.2250 | 0.6750 | 0.4620 |

Nothing clears anything. And the ΔSharpe against the MA crossover is +0.17 with
p = 0.70 - the 0.12 R per trade gap in the table above rests on 131 crossover
trades and does not survive being asked for an interval.

**Claim 1 is unproven rather than refuted**, which is a weaker verdict than the
ones on claims 2 and 3 and the honest one. The pattern across the two runs is
consistent and worth stating plainly: with the destructive overlay attached the
entry is exactly random (51st percentile, p = 0.49); with it removed the entry
is better than random but not significantly (78th percentile, p = 0.23), better
than the two alternative entry rules but not significantly, and still behind
simply holding gold over the same window (-5.3% against +30.8%).

## 5. Stability, and every parameter

Non-overlapping six-month windows, dev:

| window | V10 Sharpe | V10 ret% | B5 Sharpe | B5 ret% |
| --- | ---: | ---: | ---: | ---: |
| 2020H1 | -3.37 | -14.1 | -0.58 | -4.9 |
| 2020H2 | -0.63 | -3.4 | -0.10 | -1.8 |
| 2021H1 | -1.73 | -11.9 | -0.69 | -8.1 |
| 2021H2 | -2.35 | -12.5 | -1.19 | -11.2 |
| 2022H1 | -0.50 | -3.8 | +0.71 | +6.7 |
| 2022H2 | +0.61 | +3.7 | +1.35 | +13.5 |
| 2023H1 | -1.88 | -10.5 | -0.74 | -6.9 |
| 2023H2 | +0.48 | +2.3 | +0.94 | +7.6 |

V10 is positive in 2 of 8 windows (mean -1.17, sd 1.40); B5 in 3 of 8 (mean
-0.04, sd 0.93). B5 is not stable, but its instability is centred on zero rather
than on a loss, and the two positive stretches are 2022H2 and 2023H2.

One-factor-at-a-time sensitivity, V10, dev, ticks. The band is the realised
range, which is the honest width of the claim - not the best cell in it:

| parameter | values | Sharpe band |
| --- | --- | ---: |
| delta_SL ($) | 1.5, 2.0, **3.0**, 4.5, 6.0 | -1.87 .. -0.26 |
| delta_TP ($) | 10, 15, **20**, 30, 40 | -1.89 .. -0.66 |
| W (swing bars) | 5, 8, **10**, 14, 20 | -1.20 .. -0.59 |
| half-exit trigger (R) | 0.5, 0.75, **1.0**, 1.5, 2.0 | -1.29 .. -0.78 |

Every cell of every sweep is negative. The specification's chosen values are not
a lucky corner - they are roughly mid-band - but there is no corner to find. The
`delta_SL` sweep is the informative one: performance improves monotonically as
the stop widens from 1.5 to 4.5 dollars, which is the same arithmetic as
section 1 seen from the other side. A tight stop on gold is expensive relative
to the round turn, and the strategy's problem is that its risk unit is small.

Residual serial dependence, robust Ljung-Box on daily returns: V10 Q(10) = 16.01
(p = 0.099), B5 11.77 (p = 0.301), FixedR 19.59 (p = 0.033). Nothing here
invalidates the HAC and bootstrap inference used throughout, which is what the
test was for.

## 6. Validation, and why the test split was not spent

Read once, 1.5 years, real ticks:

| run | n | return% | Sharpe | Calmar | maxDD% | win% | payoff | PF | mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V10 half-exit on | 1,046 | -11.8 | -0.52 | -0.25 | -27.1 | 41.3 | 1.34 | 0.94 | -0.029 |
| **B5 half-exit off** | 920 | **+30.5** | **+0.67** | +0.73 | -22.2 | 15.1 | 6.14 | 1.09 | **+0.094** |

The exit result replicates and strengthens: the overlay costs about 1.2 Sharpe
here as well, and its signature is identical (win rate 41.3% against 15.1%,
payoff 1.34 against 6.14).

B5 itself is the interesting row, and it is not a candidate.

```
--- B5, validation ---
  trades             920  (522/yr)
  net return         +30.53%
  CAGR               +16.32%
  Sharpe (daily)     0.62
  max drawdown       -22.22%
  win rate           15.1%
  profit factor      1.09
  expectancy         $+33.18/trade
  edge at mid        $+57.52/trade, no costs
  all-in cost        $24.34/trade
  cost drag          42.3% of gross
  median hold        50.7 min
```

Three reasons this does not get the test split:

1. **It did not survive dev.** Sharpe -0.01 over four years and 1,666 trades.
   Nothing about the dev result would have caused anyone to nominate it. A
   strategy that is flat on the selection period and good on the validation
   period is a strategy selected by the validation period, and spending the test
   split on it would convert one honest estimate into a third bite.
2. **A profit factor of 1.09 with 42% cost drag is not a margin.** The gross
   edge is $57.52 a trade and the round turn is $24.34. A 40% error in the
   slippage assumption - which is an assumption, not a measurement - moves the
   net result by more than the whole validation return.
3. **The window flatters it.** 2024 into mid-2025 is the strongest trending
   stretch for gold in the corpus, and a 6.67:1 breakout system is precisely the
   thing that looks good in one. The dev split contains no comparable stretch,
   and the walk-forward table shows the strategy's positive windows are the
   trending ones.

**The test split remains unspent.**

## 7. What was learned that is worth keeping

Independent of gold and of this particular system:

1. **An exit overlay that raises the win rate on a high-payoff system is a
   liability, and the win rate is how you spot it.** The pairing here is
   extreme - 13.8% to 41.2%, payoff 6.18 to 1.24 - and the direction generalises
   to any rule that truncates the right tail of a positively skewed system.
2. **"Which component helps" needs a component-level ablation, and this one
   would have been reported backwards without it.** Running V10 alone gives a
   losing system and no way to know that the overlay was the loss. Running V10
   against B5 gives the answer in one number.
3. **Backtest resolution can err in either direction, and which one depends on
   the geometry.** The received wisdom is that bar-level backtests flatter. Here
   the near barrier is the stop, the candle-order convention hits it first, and
   the bar-level backtest understates by 0.038 R a trade - more than the
   strategy's entire edge.
4. **A sizing study needs the invariance check printed next to its own table.**
   Five rules, five different equity curves, and a Sharpe column that barely
   moves. Without `Sharpe(k r) = Sharpe(r)` on the same page, the return column
   would look like a result.

## 8. What would change the answer

1. **A wider stop.** The `delta_SL` sweep improves monotonically to 4.5 dollars
   and the study stops there. The specification's 3-dollar stop is small
   relative to gold's H4 ATR (8-ish dollars on dev), which is why 59% of trades
   stop out. A version anchored to ATR rather than to a fixed dollar amount is a
   different strategy and a reasonable next thing to test.
2. **B5 surviving a dev-length sample.** If the entry has anything, four years
   of dev should not be flat. A longer history, or the same rules on a second
   instrument, would settle whether validation was a regime or a signal.
3. **A measured slippage number.** Cost drag of 42% on the only positive cell
   makes the slippage model the binding assumption. It is currently
   `CostModel`'s default latency, not a measurement of this account's fills.
4. **Nothing about the overlay.** Sections 1, 2 and 6 agree across splits,
   across tapes and across the parameter sweep. That question is closed.

## 9. Reproducing

```bash
python scripts/backtests/backtest_structural_break.py                       # every axis, dev then validation
python scripts/backtests/backtest_structural_break.py --only exit           # the half-exit ablation
python scripts/backtests/backtest_structural_break.py --only sizing         # V10 vs V11, 1x and 2x
python scripts/backtests/backtest_structural_break.py --only modes          # ticks vs interp vs close
python scripts/backtests/backtest_structural_break.py --only entry --paths 1000   # the random-entry null
python scripts/backtests/backtest_structural_break.py --only walk sweep     # stability and sensitivity
python -m pytest tests/test_structural_break.py                   # pivots, tapes, overlay, sizing
```

A full tick-mode pass over dev is about seven seconds - the entry signals are
found once on H4 bars and the tape is walked once per year of ticks. The
random-entry Monte Carlo runs on the interpolated tape because a thousand
tick-mode paths would take the better part of an hour; `--paths 200` is the
default and takes a few minutes.

Inference is `src/qlab/stats.py` throughout: Newey-West standard errors,
Lo (2002)'s GMM standard error for Sharpe ratios, the Politis-Romano stationary
block bootstrap for every confidence interval and every paired difference, a
two-proportion z-test for win rates, the heteroskedasticity-robust Ljung-Box of
Lobato, Nankervis and Savin for residual dependence, and Bonferroni and
Benjamini-Hochberg corrections applied **within** the three pre-specified
families rather than across all of them at once.
