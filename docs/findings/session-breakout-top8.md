# TOP8_2026 at fixed size with a $0.30 spread guard

Subject: `XAUUSD_SessionBreakout_2026.mq5`, preset `TOP8_2026`, with
`InpUseVolTargeting = false` (a flat 0.02 lots per window) and
`InpMaxSpreadUSD = 0.30` applied literally, in dollars. XAUUSD only.

This is a follow-up to [session-breakout.md](session-breakout.md), which
rejected all four presets in R units with the spread guard translated as 3.4x
mean spread. That study never tested the dollar-sized configuration, the literal
guard, or the claim `TOP8_2026` actually rests on: **that re-picking hours on
the recent regime works.**

**Verdict: not deployable.** Two of the five pre-registered criteria fail. In
the only window neither this preset nor its parent was selected on (2020-2023),
it loses. Its dollar profit is a bet that 2026's volatility persists, because
fixed lots let dollar risk per trade grow 5.7x with the bracket width. Re-picking
hours from trailing P&L does beat picking at random, but the sets it picks still
lose in R out of sample.

Numbers come from `scripts/backtests/backtest_session_breakout_top8.py`, with
output in `reports/strategies/session_breakout_top8/summary.json`.

---

## Pre-registration

Registered 2026-09-14, before `scripts/backtests/backtest_session_breakout_top8.py`
produced a number.

**Disclosure.** Before registering, I had seen the earlier study's per-year
TOP8 figures (negative R in 2020-2022, positive from 2024). D1 and D2 are
therefore not blind to the sign of dev. They are the project's standing rules
for any strategy, not thresholds chosen for this one.

**Status of the splits.** The preset was fitted on 2026 data, which sits inside
the locked `test` split. The earlier study declared `test` spent for this
strategy family. 2026 is reported as the **fit window**, and nothing is decided
on it.

**Power.** Dev + validation is about 1,400 trading days, so t = 2 needs an
annualised daily Sharpe of about 0.85. The 2026 window is about 170 days, so it
needs about 2.4 there. A 2026-only result cannot confirm anything short of an
exceptional strategy.

### Deploy criteria and outcome

All five must hold. They are evaluated on daily dollar P&L at the deployed
size, with R reported alongside.

| id | criterion | result | |
| --- | --- | --- | --- |
| D1 | dev + validation pooled: daily net $ Newey-West t >= 2 | t = 1.00 (R: -0.77) | **FAIL** |
| D2 | neither dev nor validation alone has t <= -2 | dev -1.10, val +2.36 | pass |
| D3 | beats the flipped-direction control on dev + validation, paired block bootstrap p < 0.05 | +$4.84/day [+0.78, +9.39], p 0.017 | pass |
| D4 | walk-forward re-selection positive out of sample at t >= 2 **and** >= 95th percentile of random selection, both lookbacks | 6m: t 2.10, 98.7th; 12m: **t 1.24**, 97.1st | **FAIL** |
| D5 | dev + validation still net positive at adverse slippage 1.0 and at 1000 ms | +$1,143 / +$1,239 | pass |

The flipped-direction control takes the other side at the same instant, with the
same stop and target distances. Walk-forward re-selection re-picks 8 distinct
hours every quarter from trailing net R and trades them the next quarter, over
2021-2026.

The selection grid is hours 0-20 x brackets {15, 30, 60} min x targets
{1, 2, 3}x, 189 cells, which contains every window of all four presets. It was
checked against the preset's own tape and reproduces all 12,220 trades exactly
(max |dR| = 0).

---

## 1. The record

At 0.02 lots fixed, net of spread (crossed on the tape), commission and each
era's own measured slippage:

| period | trades | mean R | net $ | $/yr | daily t ($) | daily t (R) | max DD $ | worst day $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev 2020-23 | 7,055 | **-0.040** | -1,194 | -305 | -1.10 | **-2.11** | 1,690 | -128 |
| val 2024-25H1 | 2,901 | +0.054 | +2,926 | +1,961 | 2.36 | 1.91 | 873 | -150 |
| **dev + val pooled** | 9,956 | **-0.013** | +1,732 | +320 | **1.00** | -0.77 | 1,690 | -150 |
| test 25H2-26 | 2,264 | +0.087 | +11,799 | +10,117 | 3.09 | 2.90 | 1,108 | -693 |
| 2026 (fit window) | 1,292 | +0.147 | +11,230 | +17,019 | 3.34 | 3.62 | 797 | -693 |

The dev drawdown lasted 1,383 days, and only 44% of dev months were positive.
83% of all the dollars the strategy has ever made came in the eight months it
was fitted on.

## 2. Validation is not clean for this preset either

Five of TOP8's eight hours (0, 1, 2, 5, 14) are hours of the `ORIGINAL` preset,
which the expert's header calls "the 2024-2025 configuration". So the hour
choice has already seen most of the validation split, and its t = 2.36 there is
partly in-sample.

**2020-2023 is the only window no selection saw**, and there the preset loses
at a daily t of -2.11 in R. That is the most important number in this report.

## 3. At fixed lots, the dollar curve is a volatility bet

| year | bracket $ | risk/trade p50 | p95 | cost R | mid R | net R | net $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 5.59 | $8.83 | $27.69 | 0.092 | +0.065 | -0.027 | -466 |
| 2021 | 4.27 | $6.81 | $20.29 | 0.070 | +0.029 | -0.041 | -532 |
| 2022 | 4.63 | $7.31 | $22.27 | 0.056 | -0.032 | -0.088 | -426 |
| 2023 | 4.12 | $6.40 | $20.95 | 0.061 | +0.060 | -0.001 | +230 |
| 2024 | 5.88 | $9.31 | $28.44 | 0.047 | +0.074 | +0.027 | +675 |
| 2025 | 11.20 | $17.95 | $54.83 | 0.029 | +0.087 | +0.058 | +2,820 |
| 2026 | 22.47 | $36.64 | $109.62 | 0.021 | +0.168 | +0.147 | +11,230 |

With volatility targeting off, the stop is the bracket, so median dollar risk per
trade went from $6.40 in 2023 to $36.64 in 2026. Pooled dev + validation is
positive in dollars (+$1,732) and negative in R (-0.013) for exactly this
reason: the winning trades were the larger ones.

That is not skill. It is leverage rising automatically into the regime that
happened to pay. If gold's range contracts, dollar risk shrinks too, but only
back to the size that lost for four years. If it widens further, a single
day's loss grows with it.

## 4. The breakout direction carries a little information, less than it costs (D3)

Pooled dev + validation, the real trade averages -0.013R and its mirror averages
-0.070R. Both pay the same costs, so half the gap, **about +0.03R a trade**, is
what the direction is worth. Dev's round turn costs 0.056-0.092R. The signal is
real enough to beat its control (p = 0.017) and too small to pay for itself
before 2024. In 2026 the mirror loses -0.060R against the real trade's +0.147R.

## 5. Re-selecting hours: better than random, not good enough (D4)

Against 4,000 random 8-hour sets drawn from the same 189-cell space:

| period | TOP8 mean R | random median | random 95th | TOP8 percentile |
| --- | ---: | ---: | ---: | ---: |
| dev 2020-23 | -0.040 | -0.069 | -0.039 | 94.3 |
| val 2024-25H1 | +0.054 | -0.013 | +0.032 | 99.2 |
| 2026 (fit window) | +0.147 | -0.014 | +0.052 | 100.0 |

The preset's cells are better than a random choice from this space even in dev.
But the space itself averages -0.05R there, so "better than random" still means
losing. Cell rankings barely carry across regimes: the Spearman correlation
between 2026 H1 and pre-2026 cell ranks is 0.24. TOP8's cells rank 1, 3, 8, 13,
28, 54, 85 and 111 in 2026 H1, and 1, 8, 12, 33, 49, 59, 147 and 174 before.
`h18_r60_t3` is 174th of 189: daily t -5.66 in dev, -2.64 in validation, 0.00 in
2026.

The procedure itself, re-selecting every quarter from trailing P&L and trading
the next quarter:

| lookback | OOS mean R | OOS net $ | t ($) | t (R) | pre-2024 R (t) | 2024+ R (t) | pct vs random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 months | -0.016 | +6,926 | 2.10 | -1.03 | -0.066 (-3.38) | +0.038 (1.70) | 98.7 |
| 12 months | -0.020 | +3,677 | 1.24 | -1.33 | -0.054 (-2.74) | +0.017 (0.80) | 97.1 |

Trailing selection genuinely helps: it lifts the grid's -0.041R average to about
-0.018R, at the 97th-99th percentile of random selection. Hour effects therefore
persist for a few quarters. But the selected sets still lose in R out of sample,
lose significantly before 2024, and turn a profit in dollars only by riding the
same widening brackets as section 3.

## 6. The spread guard and cost stress

| variant | dev+val trades | mean R | net $ | t ($) | 2026 trades | 2026 net $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| guard off | 10,117 | -0.017 | +1,320 | 0.75 | 1,292 | +11,230 |
| $0.50 | 10,010 | -0.014 | +1,544 | 0.89 | 1,292 | +11,230 |
| **$0.30** | 9,956 | -0.013 | +1,732 | 1.00 | 1,292 | +11,230 |
| $0.20 | 9,581 | -0.008 | +2,199 | 1.28 | 1,286 | +11,108 |
| 3.4x mean spread | 9,956 | -0.013 | +1,707 | 0.99 | 1,292 | +11,230 |
| slippage adverse 0 | 9,956 | -0.005 | +2,357 | 1.36 | 1,292 | +11,681 |
| slippage adverse 1.0 | 9,956 | -0.020 | +1,143 | 0.66 | 1,292 | +10,890 |
| 1000 ms latency | 9,956 | -0.019 | +1,239 | 0.72 | 1,292 | +10,956 |

**The $0.30 guard never binds in 2026**: gold's spread there sits below it, and
the 2026 row is identical with the guard off. It removed 161 dev and validation
fills worth about $400. It is harmless, and it does nothing in the regime the
preset is meant for. It also reproduces the earlier study's 3.4x-mean
translation almost exactly.

No realistic cost assumption changes the sign in either direction.

## 7. What it looks like on an account at 0.02 lots

* **Concurrency.** Up to 6 positions open at once, all on the same side, with
  $763 at risk simultaneously. At entry, 4 positions are open at the 99th
  percentile. `InpMaxOpenPositions = 8` never binds.
* **Worst day** -$693, worst month -$533, both from the high-volatility years.
* **2026 by month:** +$2,778, +$1,414, +$2,818, +$855, +$834, +$1,197, +$1,031,
  then **August +$302 at +0.004R**. That is one month and proves nothing, but it
  is the most recent data there is.

## 8. The MT5 file

The copy in the terminal's `MQL5/Experts` folder (modified 2026-08-27) is
**older than `mt5/XAUUSD_SessionBreakout_2026.mq5`** (2026-09-10). Its trading
logic is identical, but it lacks two fixes:

* `AdoptExistingState()`. Without it, a terminal restart, a recompile or an
  input edit while brackets are armed re-arms each window and stacks a second
  bracket at the same prices. Both then fill on the break, at double the
  intended size.
* The persisted daily-equity baseline. Without it, the daily loss halt measures
  from the last restart rather than from the start of the day.

`InpMaxDailyLossPct` also defaults to 0 (off). Every other defect listed in
[session-breakout.md](session-breakout.md) section 7 still applies.

## 9. If it is to be pursued

This is a bet that the 2024-2026 regime persists, and nothing in the corpus can
confirm or refute it any further. The only honest evidence left is forward from
2026-09-01.

At validation's Sharpe of 1.9, reaching t = 2 takes about 270 sessions, around 13
months. At 2026's 3.9 it takes about 65. At a Sharpe of 1.0 it takes four years.
Two things follow for any forward run:

* Judge it in R, not dollars. Dollars will rise with volatility whether or not
  there is an edge.
* Fix the hours before it starts. Re-picking them mid-run restarts the clock.

## 10. The MT5 Strategy Tester report

Added 2026-09-15. A Strategy Tester report of the same expert
(`ReportTester-434037516.xlsx`, Exness trial server, M1 "18% real ticks") looks
deployable on its own terms:

* **Settings.** `InpPreset=2` (TOP8_2026), 0.02 lots, targeting off, $0.30
  spread cap, **`InpMaxDailyLossPct=4.5`**, $1,000 deposit, 2023-01-02 to
  2026-09-09.
* **Headline.** +$16,631, 7,022 trades, profit factor 1.25, MT5 Sharpe 2.41,
  maximal balance drawdown $1,215.

`scripts/research/tester_reconcile.py` pairs the report's 14,044 deals into
trades. Every close reproduces its printed profit to within $0.008. It then
compares them with the tick model and replays the report's account rules on
the model.

**The report and the model are the same backtest.** Over 2023-01-02 to
2026-08-31:

* The tester has 6,980 trades and the model 6,994. 6,941 match on day and
  window.
* 99.45% of matched trades go the same direction and 99.4% exit the same way
  (stop, target, or expert close). The median entry-time difference is 8
  seconds.
* The tester's 211.5M ticks are close to the corpus's 209.1M real ones.

The report is not new evidence. It is this study's trades, over a window that
starts where the losses stop.

**Where the tester's extra +$2,003 comes from** (tester $16,958, model $14,955):

| source | $ |
| --- | ---: |
| same trades, same exit: commission, entry and exit fills | +1,827 |
| of which: commission on the entry deal only ($0.07 vs a $0.14 round turn) | +436 |
| of which: entries with no slippage (tester fills 0.02 over mid, model 0.05) | +634 |
| closes at the reopen after the daily halt (section below) | +528 |
| daily-loss-halt closes | +320 |
| trades in opposite directions (38) | +388 |
| holds over a weekend or holiday | -91 |
| other exit disagreements | +3 |
| trades only the tester took | -29 |
| trades only the model took | -943 |

So the tester's gain over the model is its cost assumptions: roughly half the
commission and no slippage. The remaining differences net out.

**With the report's own account rules on the model** ($1,000, 0.02 lots, 4.5%
daily halt, 2023-01-02 to 2026-08-31): $1,000 becomes $14,776, with a maximum
drawdown of $1,087 (44.7%) and 44 halt days. The tester shows $17,958 and 31.2%
over the same dates. The halt costs $1,179 against no halt.

**The start date decides the report.** Same rules, same model, run to
2026-08-31:

| start | end balance | net before 2026 | lowest balance | max DD | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 2020-01-29 | $0 | -$1,000 | $0 | 100% | **wiped out 2022-03-28** |
| 2021-01-01 | $0 | -$1,000 | $0 | 100% | **wiped out 2024-04-19** |
| 2022-01-01 | $12,020 | +$1,457 | $368 | 74.5% | survives |
| **2023-01-02 (the report)** | $14,776 | +$3,087 | $945 | 44.7% | best start |
| 2024-01-01 | $12,455 | +$2,334 | $836 | 49.3% | |
| 2025-07-01 | $10,845 | **-$180** | $562 | 51.4% | |
| 2026-01-01 | $10,907 | - | $871 | 27.5% | |

The daily halt does not prevent ruin. At fixed lots a single stop costs $6-28
whatever the balance, so a shrinking account simply halts earlier each day.

In the report's own numbers, 71% of the profit to 2026-08-31 came in 2026:
2023 +$520, 2024 +$1,020, 2025 +$3,408, 2026 +$12,011. That is the window the
preset was tuned on. The report's nine days after the corpus ends (2026-09-01
to 09-08) are 42 trades and -$327. That is too few to mean anything, and it
is not a good sign either.

**Two things the report exposes about the expert.**

1. **In US winter, the tester never executed the weekday flatten.** It fired
   on 0 of 232 weekday winter days. On 222 of those days the ticks show gold
   still quoting at 21:53, yet positions stayed open to the 23:00-23:05
   reopen. The only winter closes near the flatten were Friday 20:00 closes
   (57). In summer the flatten worked on 600 of 613 days. The likeliest cause
   is the tester applying the symbol's current (summer) trading sessions all
   year, rejecting closes after 20:58. It is harmless if the live server
   updates its sessions for winter, and a nightly unhedged hold through the
   halt if it does not. **Check the XAUUSD trade session on the live account
   in November, before relying on the flatten.**
2. **On early-close holidays, positions are carried over the weekend.** On
   Black Friday, Christmas Eve, New Year's Eve, July 3/4 and Juneteenth, gold
   halts between 17:00 and 19:30 UTC, before the Friday 20:00 close and the
   flatten. The flatten is tick-driven, so nothing fires, and 22 positions
   were held for up to 69 hours until the Sunday reopen. This happens live
   too. The fix is to flatten at a clock time the market is known to still
   be quoting, or on the last tick before any quote gap.

The verdict above stands. The report reproduces the backtest that was already
rejected, under cheaper costs, starting on the most favourable date.

## 11. Why the Deploy-build tester report made less

Added 2026-09-15. A second report, `ReportTester-434037516-dploy.xlsx`, ran
`XAUUSD_SessionBreakout_Deploy` over the same dates and made **+$9,515**, where
the first made +$16,631. Its maximum relative drawdown was 57%, against 31%.

It trades the same windows: in the Deploy build's enum, `InpPreset=3` is
`PRESET_TOP8_2026`. Four settings differ:
- risk sizing at **2% of equity per trade**, where the build's own default is 0.25%
- `InpMinRangePct` 0.05 -> **0.25**
- daily loss halt 4.5% -> **3%**
- `InpMaxOpenPositions` 8 -> 7

`scripts/research/tester_compare.py` matches the two reports' trades and
changes one thing at a time:

| step | $ |
| --- | ---: |
| first report | +16,631 |
| trades skipped under the 0.25% range floor (3,735) | -1,880 |
| trades skipped after the 3% daily halt had fired (620) | -3,782 |
| wide brackets skipped because 2% sized under 0.01 lots (74) | -4,018 |
| same trades, closed early by the daily halt (425) | -2,258 |
| same trades, not halted (2,168): tick-data differences | -21 |
| 2%-of-equity lots instead of a fixed 0.02 | +4,447 |
| trades only the Deploy run took (18) | +397 |
| **Deploy report** | **+9,515** |

* **The daily halt is most of it: -$6,040.** At 2% risk, two stops take the
  account through a 3% limit. The halt fired on 428 trades, against 21 in the
  first report. This strategy wins 36-39% of the time and pays through
  occasional 3R targets, so it expects a string of small losses before a winner. A
  tight daily stop cuts off exactly the part of the day that pays. The trades
  it prevented averaged +0.12R.
* **The minimum lot cost the best trades: -$4,018.** The Deploy run was deep
  in its drawdown in early 2026: its balance fell from $5,006 in May 2025 to
  $2,157 on 2026-01-28. On a ~$2,400 balance, 2% risk on a $60-100 bracket
  sizes to 0.005-0.008 lots. That is below the 0.01 minimum, so the EA skips
  rather than over-risks. Those 74 trades were January-March 2026's widest
  brackets, averaging +0.41R.
* **The range floor removed trades that were profitable in the tester:
  -$1,880, +0.05R each.** The Deploy header justified 0.25% on the 60-minute
  brackets of `PRESET_DEPLOY`. TOP8 arms four 30-minute windows, and the floor
  removes 74-83% of their trades.
* **Tick quality is not a factor.** On the 2,168 matched trades the halt did
  not touch, the 18%-real and 98%-real runs differ by $21.
* **Larger size bought the dollars back, and doubled the risk.** Median risk
  was 1.6-1.8% of the balance per trade, against 0.27-0.54% at a fixed 0.02
  lots. With up to seven positions open, that produced the 57% drawdown.

None of this changes the verdict. Both reports trade the same preset over the
window it was tuned on, and a daily stop tighter than two trades' risk is not
compatible with a positive-skew strategy.

## Reproducing

```bash
python scripts/research/tester_reconcile.py ReportTester-434037516.xlsx --phase parse halt report detail
python scripts/research/tester_reconcile.py ReportTester-434037516-dploy.xlsx --phase parse --out reports/strategies/session_breakout_top8/tester_deploy
python scripts/research/tester_compare.py reports/strategies/session_breakout_top8/tester reports/strategies/session_breakout_top8/tester_deploy
python scripts/backtests/backtest_session_breakout_top8.py              # tapes, grid, analysis (~10 min, <1 GB)
python scripts/backtests/backtest_session_breakout_top8.py --phase analyse
python -m pytest tests/test_session_breakout.py tests/test_session_breakout_top8.py tests/test_session_breakout_halt.py -q
```
