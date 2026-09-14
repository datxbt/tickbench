# Findings

Twenty-nine write-ups, each a hypothesis tested on 696M ticks of Exness raw-spread
data (EURUSD, USDJPY, XAUUSD, USTEC, 2020-2026) against the measured cost model
in [../cost-model.md](../cost-model.md). **One strategy was accepted and one
forecasting model half-accepted; everything else was rejected.**

The table is ordered by verdict. The summaries below it are the shorter version
of each write-up, in the order the studies were run.

| Study | Verdict |
| --- | --- |
| [USTEC risk-managed long](ustec-risk-managed-long.md) | **accepted**, with a narrow claim |
| [Forecasting intraday volume](news-breakout-and-volume-spar.md) | **half accepted** - the forecast is real, the money is not |
| [Intraday momentum on USTEC](intraday-momentum-and-btc-dynamics.md) | replicates in mechanism, decays to zero out of sample |
| [The 5-minute opening range breakout](opening-range-breakout.md) | does not carry to validation |
| [The 15-minute opening range, pre-registered](orb15.md) | inconclusive on the held-out split, not deployable |
| [Sizing for a two-step prop challenge](propfirm-sizing.md) | sizing study for the accepted overlay |
| [Least time to a funded account](propfirm-speed.md) | sizing study for the accepted overlay |
| [XAUUSD](xauusd-rejected.md) | **rejected**, no strategy |
| [Gold price levels](level-interaction.md) | **rejected** |
| [Round numbers on EURUSD, USDJPY and USTEC](osler-round-numbers.md) | **rejected** |
| [Discovery program: tail-conditioned structure](discovery-program.md) | **rejected**, 16 hypotheses; minute-scale reversal is real in 2020-2023 and gone in 2024-2025 |
| [Rebalancing flow into the US close](close-rebalance.md) | **rejected** |
| [USDJPY carry harvest](usdjpy-rejected.md) | **rejected** on the held-out split |
| [Scheduled flows: macro releases and the FX fixings](scheduled-flows.md) | **rejected** on the held-out split |
| [EURUSD](eurusd-rejected.md) | **rejected**, no strategy |
| [Big bar, pause bar, break of the pause bar](pause-bar.md) | **rejected** |
| [The engulfing candle](engulfing.md) | **rejected** |
| [The engulfing bar's quadrants](engulfing-quadrant.md) | **rejected** |
| [Liquidity sweeps and order blocks (UAlgo)](sweep-orderblock.md) | **rejected**, both setups, the opposite-block target and the reversed trade, 1m to D1 |
| [Precision Sniper confluence scoring (WillyAlgoTrader)](precision-sniper.md) | **rejected**, all 6 presets, 1m to D1, 44 exits; matches a coin flip on a random bar |
| [The New York open EMA rule](open-ema.md) | **rejected** - the stated trailing exit matches a coin flip, and the reported 57% win rate is not reachable by any trailing stop |
| [Session opening-range breakout](session-breakout.md) | **rejected** |
| [The overnight-intraday reversal family](overnight-reversal.md) | **rejected** |
| [Gold structural breaks](structural-break.md) | **rejected** |
| [The overnight drift](overnight-drift.md) | **rejected** |
| [Gold VWAP/EMA regime filter](vwap-ema-gold.md) | **rejected** |
| [Decision trees on next-bar direction](decision-tree-intraday.md) | **rejected** |
| [Small-cap retail strategies](smallcap-poudel.md) | **rejected** |
| [Machine-learned FX signals](fx-ml-enkhbayar.md) | **rejected** |

Two of the write-ups evaluate two papers each, so the table has fewer rows
than the studies it covers.

---

## Sizing the USTEC overlay for a two-step prop challenge

Full report: `docs/findings/propfirm-sizing.md`. Rules engine in
`qlab.propfirm`; the strategy itself is unchanged and nothing about it was
fitted here - what is chosen is a **size**, on dev and validation.

A challenge is a first-passage problem, not a Sharpe problem: reach +10% then
+5% without touching a 5% daily floor or a **static** 10% total floor. Three
consequences, each of which changes what "good" means:

- **Size is the whole decision.** P(touch +a before -b) is governed by
  `theta = 2*mu/sigma^2`, and scaling a strategy by `k` scales theta by `1/k`.
  Halving the size roughly doubles theta. The only thing size buys is speed -
  and FTMO has no deadline.
- **The daily floor watches floating equity, so a close-to-close backtest cannot
  see it.** A day that dips 6% and closes flat has failed the challenge and
  passed the backtest. Every figure here uses a per-day worst excursion built
  from minute bars.
- **The total floor is static, measured from the initial balance.** An account at
  +6% is 16% from the floor, not 10%. `cushion_size` spends that: size in
  proportion to the room that is left, so the position goes to zero as the
  cushion does.

Cushion sizing dominates flat sizing everywhere, and changes the failure mode:

| cushion 0.50x, strategy capped at 1.0x weight | dev | validation | test |
| --- | ---: | ---: | ---: |
| P(pass both steps) | **100.0%** | **100.0%** | **99.2%** |
| flat 0.50x, for comparison | 90.1% | 83.5% | 95.9% |
| median time to funded | 3.6 yr | 2.1 yr | 2.5 yr |
| total-loss failures | none | none | none |

**And the stress test is the most useful thing it produced.** The overlay's mean
session return is not statistically significant (t = 1.87 / 1.10 / 1.22), so the
same simulation was re-run with the drift removed and everything else kept. With
no edge, cushion sizing **almost never fails - it just never finishes**: 12-22%
complete within twelve years, the rest grind sideways above the floor. So
**cushion sizing converts risk of ruin into risk of never completing**, which for
a deadline-free challenge is a good trade - but it also means a high pass
probability is not evidence the strategy works. It is what the sizing rule does
to any series with roughly this risk.

**On speed.** Passing *one* challenge inside three months with high probability
is not reachable, and the reason is arithmetic. Passing +10% then +5% is 15.5%
compounded, so a 63-day median needs about 62% a year; the daily floor then caps
volatility near 27% if a 5% day is to stay a 3-sigma event. That is a **Sharpe
around 2.0 against the 0.95 available**, and no scaling closes it - scaling moves
`mu` and `sigma` together and leaves the ratio where it was.

**Reaching a funded account inside three months is a different question, and the
answer is yes.** See the next section: a challenge that fails is an entry fee and
a few weeks, not the end of the project, and large sizes resolve quickly in both
directions. Three months is bought with fees rather than with edge.

## Least time to a funded account

Full report: `docs/findings/propfirm-speed.md`. Driver:
`scripts/research/propfirm_speed.py`.

The report above optimises *P(pass)*. This one optimises *calendar time to the
first funded account, counting the attempts that fail*, which is what someone
pursuing a funded account actually experiences. The ranking inverts.

**Recommendation: flat 4.0x with a -3% daily breaker, flat across the weekend.**
Median **2.2 months on validation, 3.7 on dev**, about 3 entry fees, 31-37% pass
per attempt. Then drop to 1.0-1.5x the moment the account is funded - at 4x the
breaker fires 45-75 times a year and would destroy it.

Two errors in the earlier analysis had to be fixed first, both immaterial at
0.50x and decisive at 4x:

- **The reported statistic conditioned on winning.** `median_days_when_passed` is
  the time taken by runs that worked. `time_to_funded` walks attempt after
  attempt until one sticks, charging days for the failures.
- **A circuit breaker was assumed able to stop a gap.** The daily record measured
  each broker day from its own first quote, dropping the nightly halt and the
  weekend from both the return *and* the worst excursion - and then let the
  breaker truncate that move as if a stop could sit inside a closed market.

That second fix produces the finding the plan turns on: **the ceiling on size is
set by the worst reopen, not by volatility.** Holding through weekends caps
USTEC at 1.35x on dev; closing before them raises it to 4.92x.

**And the control matters more than the headline.** With the drift removed and
everything else kept, a no-edge series reaches a funded account in a median of
3.0-4.5 months at 4x, against 2.3-3.7 for the real thing. At 1x the edge is worth
+38 to +50 points of pass rate; at 4x, +10 to +14. **The fee buys compressed
time, not a better chance** - which is why the size drops the day the account is
funded, because there the edge is the entire product.

Also searched and not found: unconditional time-of-day drift, conditional
intraday predictability (session-so-far and overnight gap, eight decision times,
four symbols - the one effect that looks real in dev reverses sign in
validation), and any volatility forecast that shrinks the excursion tail. The
edge is Sharpe 0.95 and there is no more of it in this corpus.

What does help at speed is a **daily circuit breaker** (`InpDailyStopPct`):
above 1.0x every failure is the daily floor, which cushion sizing cannot touch
because that floor resets nightly. Flattening at -3% and standing down until
tomorrow lifts the pass rate at 2.0x from 15.6-34.0% to 80.1-100%. Even so, the
fastest defensible configuration has a median of **7 to 23 months to funded**
once failed attempts are counted. Getting to three months needs roughly four to
five uncorrelated strategies rather than one - the same breadth constraint the
XAUUSD study reached from the other direction.

The expert takes `InpPropMode`, `InpPropScale`, `InpPropMaxLossPct` and
`InpDailyStopPct`.

## XAUUSD - **rejected, no strategy**

Ten hypothesis families, independent of the session breakout, all failed. Full
report: `docs/findings/xauusd-rejected.md`. **Gold's test split is unspent** -
nothing survived dev and validation, so there was nothing to take to it.

**Gold's dev return is not gold's return.** `rollover.drift_decomposition` splits
a price series into the roll hours, the gaps no contiguous minute covers, and
everything else - and only the last is reachable by a strategy that goes home
flat every night:

| symbol | split | total % | roll hrs | gaps | **liquid hrs** | **liquid %/yr** |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **XAUUSD** | dev | +27.4 | +14.1 | +11.4 | **+2.0** | **+0.42** |
| **XAUUSD** | validation | +47.3 | +7.9 | +3.6 | **+35.8** | **+19.39** |
| USTEC | dev | +61.6 | +3.2 | +0.9 | +57.5 | +11.92 |
| EURUSD | dev | +0.1 | +9.1 | -1.9 | -7.1 | -1.46 |

**93% of gold's dev return sits in the roll and the halt gap** - the financing
adjustment a long earns in the price and hands back in swap - against 7% of
USTEC's over the same window. Gold's overnight basis is +2.56 bps a night, the
largest of the four, and charging it turns dev buy-and-hold from **+4.48% into
-4.47% a year**. Gold over this corpus is two instruments: a financing vehicle
in 2020-2023 and a trending asset in 2024-2025.

Two results worth carrying forward:

- **The LBMA fix was the best candidate in the whole project, and it is the
  spread.** Gold drifts down through the 10:30 London auction in both splits - a
  scheduled, published, one-sided event, the only cross-split-consistent
  directional signal found anywhere here that is not the carry roll. At the mid
  it is +0.98 bps (t 2.68). At real bid/ask fills plus commission it is
  **+0.02 bps**, and on dev that is 2020 by itself.
- **The overlay does not merely fail on gold, it destroys value**: the trend gate
  returns **-4.65%/yr against buy-and-hold's +4.48%** in dev, with a *worse*
  drawdown. Not a cost problem - 14.7x turnover at 1.05 bps is 7 bps a year - a
  timing one. Gold chopped across its 200-session average for four years. The
  general lesson across all four instruments: a volatility-sizing overlay needs
  its host to have a drift **in the hours you can hold it**, and a trend worth
  gating. USTEC has both; gold has neither in dev and both in validation.

## Gold price levels - **rejected**

The eleventh gold hypothesis family, and the first with a peer-reviewed
order-book mechanism behind it. Osler (2003, 2005) found take-profit orders
clustering *at* round numbers and stop-loss orders *just beyond* them, which
predicts reversion at a level that holds and a cascade through one that breaks.
Implemented in `qlab.levels` and measured with `qlab.eventstudy`. Full report:
`docs/findings/level-interaction.md`.

Three kinds of interaction - `touch`, `sweep`, `breach` - forming a partition, so
one approach to one level cannot be counted twice. Seven level families, six
horizons, 5m and 1m, dev and validation. Dev, net of the measured cost stack:

| kind | 30 min | 120 min | | bps/day at 30 min | t/day |
| --- | ---: | ---: | :-- | ---: | ---: |
| touch | -1.73 | -1.17 | | -4.06 | -3.11 |
| sweep | -1.78 | **-2.56** | | -5.20 | -4.05 |
| breach | -0.98 | **+0.41** | | -5.64 | -2.07 |

Four findings, in order of weight:

- **The reversion half of the mechanism is backwards on gold, and it is backwards
  at the mid.** Touch and sweep lose before any cost is charged, at almost every
  horizon, in both splits. The "liquidity sweep" - a marginal new extreme
  followed by a reversal, the most-taught intraday gold pattern there is - costs
  5.2 bps a day on dev. It is not a cost problem; it points the wrong way.
- **The breach half orders exactly as predicted, and is still the drift.** At 120
  minutes the round-number breach runs +6.27 bps at the mid on the $100 grid,
  +5.93 on $50 and +3.36 on $10 - coarser grid, larger effect, an ordering
  written into the script before the numbers were seen - while previous-day and
  previous-session extremes contribute nothing. It also sits outside the range of
  twelve placebo draws. But split by direction it is **+2.03 bps long against
  -0.59 short in validation**, which is gold's bull regime showing through a
  level definition. In dev, where gold has +0.42%/yr in holdable hours, both
  sides are flat.
- **Event-level t-stats are mostly overlap.** Breaches arrive three to a day and
  thirteen on a trending day, and events-per-day correlates +0.32 with that day's
  return - so an event-weighted mean overweights the days the signal was always
  going to work. Summed within days and tested across days, **nothing in the
  study clears t = 2**, in either split, either direction, any horizon.
- **207 cells, zero survivors.** A Sidak correction across the cells inspected
  demands |t| >= 3.66. One cell clears an uncorrected +2 (chance alone gives
  five); six clear the corrected threshold on the *losing* side.

At 1m every net cell is negative, which repeats the `pause_bar` lesson: gold's
wide bar range does not help when the predictable fraction of it is small.

The durable output is the two modules. `qlab.eventstudy` sits *before* the
engine on purpose - it maps signal to forward-return distribution rather than
signal to backtest, so no exit is chosen while an effect is still being
measured - and it reports every result three times: at the mid, at real bid/ask
fills plus commission, and net of measured slippage.

**Gold's test split is still unspent.** Eleven families rejected without it.

## USDJPY carry harvest - **rejected on the held-out split**

The USTEC overlay pointed at the yen. Full report:
`docs/findings/usdjpy-rejected.md`.

| | dev | validation | test (one run) |
| --- | ---: | ---: | ---: |
| strategy Sharpe | **0.81** | **0.26** | **0.01** |
| strategy CAGR | 4.93% | 1.40% | -0.11% |
| buy & hold Sharpe | 0.67 | 0.09 | **1.08** |
| buy & hold CAGR | 6.16% | 0.41% | **8.59%** |

A monotone decay from 0.81 to 0.26 to 0.01 is what an effect that was never
there looks like against three successive periods. On test the strategy returned
nothing while holding returned 8.59%, at the same drawdown.

Three things worth carrying forward:

- **A volatility target is not portable across volatility levels.** USTEC runs
  at 27% against a 15% target so the rule *shrinks* the position, which is where
  its drawdown reduction comes from. USDJPY runs at 9.6%, so the identical rule
  *levers* to 1.5-1.7x and makes the drawdown **worse than holding** (-19.2% vs
  -15.6% on dev). Capping the weight at 1.0 fixes that and was the starting
  point for the candidate set here.
- **The cross-asset gate failed on the one day it existed for.** Gating the yen
  position on equity trend has the best story of any candidate and the best dev
  Calmar (0.60). In August 2024 the Nasdaq fell hard but did not break its
  200-session average until after the yen had unwound, so the gate lost -13.44%
  against buy-and-hold's -13.44% - no protection at all.
- **The two candidates that top the test table are the two that failed
  validation.** Selecting on test would have picked the one with zero measured
  crash protection. That is the circularity the split exists to prevent, visible
  in a single table.

The refactor is the durable output: `risk_managed_long` now takes a
`SessionSpec`, shipping `US_CASH` and `FX_DAY`. On FX the day boundary and a
sane execution time are different instants - 17:00 New York ends the FX day and
is also the rollover, whose spread runs 1.5 bps against a weekday mean near
0.015 - so `FX_DAY` reads that close and trades at 08:00 the next morning.

## EURUSD - **rejected, no strategy**

Twelve hypothesis families, all of them failed. Full report:
`docs/findings/eurusd-rejected.md`, or the [study page](https://claude.ai/code/artifact/79a2c46e-c204-4c86-a3c9-d77b5bdbc1d2);
every rejection re-runs from `scripts/research/eurusd_research.py`.

Two structural facts, either of which would have been enough:

**EURUSD has no risk premium to fall back on.** Over dev it returned **+0.25% in
four years** - a CAGR of 0.06% at a Sharpe of 0.01. The USTEC study could end in
an accepted strategy because there was a positive-drift asset to hold and the
work was in sizing it. There is nothing here to size, so a EURUSD strategy has
to be pure alpha, and none of the twelve families produced any.

**Per unit of the volatility a strategy has to beat, the Stage 2 cost ranking
inverts:**

| symbol | round turn (bps) | 1m sd (bps) | rt / 1m sd | rt / daily sd |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | 0.585 | 1.406 | 0.42 | 0.0119 |
| USDJPY | 0.724 | 1.501 | **0.48** | **0.0122** |
| XAUUSD | 1.092 | 2.704 | 0.40 | 0.0113 |
| USTEC | 1.209 | 4.493 | **0.27** | **0.0072** |

In bps EURUSD is the cheapest instrument and USTEC the dearest, which is what
Stage 2 reported. Divided by the move that has to pay for it, the order reverses
and the spread widens: USTEC is **1.6x cheaper** than EURUSD either way you
normalise, because commission is a fixed 0.455 bps and EURUSD moves a third as
much as an index does. Cost in bps is not the number that decides what to trade.

**The one finding worth keeping is a measurement, not a strategy.** A long held
across the broker's daily roll earns +1.19 bps at tradable prices, t = 5.05, and
the sign is stable across both splits - the only thing in the study that was.
It is the carry. Spot settles T+2, so the quote adjusts by the tom-next forward
points at each roll and the broker charges swap to offset it. The test that
settles it is the sign on instruments whose carry runs opposite ways:

| symbol | carry on a long | predicted | dev | validation |
| --- | --- | --- | ---: | ---: |
| EURUSD | pays | up | +1.35 (t 5.05) | +0.97 (t 1.76) |
| USDJPY | **receives** | **down** | **-1.47** (t -5.89) | **-1.44** (t -2.51) |
| XAUUSD | pays | up | +2.56 (t 3.85) | +2.95 (t 2.38) |

Six of six. `qlab.rollover` computes this for any symbol at tradable prices and
`basis_table()` runs the sign test; Stage 2 listed overnight swap under "not
modelled", and this narrows that gap to the charge alone.

## USTEC risk-managed long - **accepted, with a narrow claim**

`USTEC_RiskManagedLong.mq5`, modelled in `qlab.strategies.risk_managed_long`.
Full report: `docs/findings/ustec-risk-managed-long.md`, or the
[study page](https://claude.ai/code/artifact/e4202817-fa9b-4031-a163-c8e4767804d9).

**It has no directional edge and does not claim one.** Sixteen directional
hypotheses were tested on USTEC first and all sixteen failed; the rejection log
is the bulk of the report. What is deployed sizes a long position by a
volatility forecast, because volatility is forecastable here even though
direction is not - on non-overlapping 20-session blocks, one block's volatility
predicts the next at +0.45 / +0.23 / +0.15 across the three splits, while one
block's *return* predicts the next at -0.05 / -0.15 / +0.08.

| | dev | validation | test (one run) |
| --- | ---: | ---: | ---: |
| CAGR | 11.14% | 13.77% | 18.87% |
| Sharpe | 0.95 | 0.91 | 1.15 |
| max drawdown | **-15.1%** | **-12.9%** | **-7.9%** |
| Calmar | **0.74** | **1.07** | **2.40** |
| buy & hold Sharpe | 0.60 | 0.90 | 1.20 |
| buy & hold max drawdown | -39.8% | -24.9% | -10.8% |

Same risk-adjusted return as a passive long, **roughly half the drawdown**, in
all three periods. It loses to buy and hold on raw return in all three, and the
mean session return is not statistically distinguishable from zero (t = 1.87 /
1.10 / 1.22) - as it cannot be, on a few hundred sessions.

Three results from the rejection log are worth carrying forward:

- **The broker changed USTEC's session hours in June 2023, and it killed a real
  anomaly.** The gap across the daily halt was worth +5.44 bps at t=4.09 while
  the halt ran 16:00-18:30 New York and spanned an hour in which NQ futures kept
  trading. Since the halt moved to 17:00-18:00 - exactly the CME maintenance
  break - it is worth +0.11 bps at t=0.09. A whole-corpus backtest would have
  found a t=4 edge and deployed into a market that stopped offering it. Session
  boundaries belong in New York local time, not UTC: the broker has moved the
  UTC hours twice and the New York hours never.
- **Intraday seasonality on USTEC is noise.** The dev-versus-validation
  correlation of the average intraday drift shape across 46 half-hour windows is
  **-0.11**, and a selection made on dev scores Sharpe +0.81 in sample and
  **-1.79** out of sample.
- **Minute-frequency predictability is real and about 20x too small to trade.**
  The strongest relationship in 1.3M minute bars is 60-minute reversal at t=-29;
  its average conditional edge is 0.56 bps against a 1.40 bps round turn, and a
  non-overlapping event study finds no gross edge at all. The best cross-asset
  model over all four symbols predicts 0.047 bps and clears cost on 0.01% of
  minutes.

The one cost this corpus cannot measure is **overnight swap** - a quote feed
carries no financing. It is an explicit input, defaulting to zero, and the report
states the break-even (4.8-6.1 bps/night) rather than burying an assumption.

## Big bar, pause bar, break of the pause bar - **rejected**

The price-action classic: a big bar breaks out of resistance, the next bar is a
very small pause bar, you enter on the break of the pause bar and put the stop
under it - said to work on all assets and all timeframes. Implemented in
`qlab.strategies.pause_bar` and run tick-by-tick on all four instruments at 1m,
5m, 15m and 1h. Full report: `docs/findings/pause-bar.md`.

149,473 signals, 77,432 filled trades. Nine exit geometries resolved against
identical fills, because steps 1-4 fix the entry and the risk and say nothing
about the way out. Mean R net of all costs, dev, at the best exit for each cell:

| | 1m | 5m | 15m | 1h |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | -0.430 | -0.073 | -0.070 | +0.153 |
| USDJPY | -0.418 | +0.002 | -0.020 | -0.005 |
| XAUUSD | -0.827 | -0.279 | -0.092 | +0.059 |
| USTEC | -0.735 | -0.168 | -0.024 | +0.286 |

Of 160 (instrument x timeframe x exit) cells on dev, **zero** clear t = +2 and
87 clear t = -2. Four findings:

- **Step 2 sets the cost.** The pause bar is selected for being small, the stop
  goes under it, so 1R is small by construction and a fixed round turn is a
  large fraction of it: **0.52-0.66R on 1m**, falling to 0.03-0.05R at 1h. The
  median 1m risk is 1.0-2.1 bps against a 0.5-0.7 bps round turn. Those are the
  same number.
- **The loss is exactly the cost.** The same trades priced at mid, less the
  measured cost, reproduce the realised result to within 0.02R at every
  timeframe. There is no execution subtlety left to find.
- **There is no edge before costs either.** The "at mid" figure is positive, but
  **fading every signal is positive at mid too, by as much or more** - it sums
  to twice itself instead of to zero. What that column measures is a property of
  pricing a stop-and-target trade at mid, not an edge. The one measure with no
  barrier in it - unconditional forward return in bps - is insignificant in
  seven of eight cells and negative in the eighth.
- **The breakout condition does no work.** Keeping the entry geometry and
  dropping step 1 entirely gives a *higher* cost-free result (+0.072R against
  +0.065R) on seven times the sample.

Twenty-six parameter settings were tried one at a time and all lose. Two point
the same way: demanding a true inside bar - the textbook version of step 2 -
takes mean R from -0.066 to -0.149, while the only setting to reach zero is the
one that selects breakout bars big enough that the pause bar after them is not
small. Sorting trades by their own risk says it again - the cost-free column is
flat across risk quintiles and only the cost moves. **The cure for this strategy
is to stop choosing small pause bars, which is to stop trading the setup.**

The test split was not spent: nothing reached it.

## The engulfing candle - **rejected**

Body-only engulfing: a candle whose body swallows the previous candle's body,
entered at market on its close, stop at that same candle's extreme, 2R target,
reversing when the opposite pattern prints, 0.01 lot. Implemented in
`qlab.strategies.engulfing` and run tick-by-tick on all four instruments across
**eleven timeframes from 1m to 4h**. Full report:
`docs/findings/engulfing.md`.

1,827,321 signals, 1,576,016 trades. Twelve exits resolved against identical
fills - targets from 0.5R to 6R plus a no-target variant - because the idea
nominates 2R only provisionally. Mean R net of all costs at the stated 2R
target, dev:

| | 1m | 5m | 30m | 1h | 4h |
| --- | ---: | ---: | ---: | ---: | ---: |
| EURUSD | -0.794 | -0.456 | -0.217 | -0.163 | -0.046 |
| USDJPY | -1.228 | -0.605 | -0.230 | -0.130 | -0.046 |
| XAUUSD | -0.631 | -0.277 | -0.122 | -0.111 | -0.007 |
| USTEC | -0.557 | -0.250 | -0.085 | -0.005 | -0.026 |

Of 528 (instrument x timeframe x exit) cells on dev, **zero** clear t = +2 and
435 sit at or below t = -2. Four findings:

- **The stop is the candle, so the cost is the timeframe.** 1R is one bar's
  range by construction, so it shrinks with the timeframe while the round turn
  does not: **0.37-0.52R on 1m**, falling to 0.020-0.031R at 4h. Notably this is
  ~3/4 of the pause bar's cost/R - the engulfing candle selects for a *large*
  stop, and is the better setup on exactly the dimension that killed the other
  one. It still loses.
- **The loss is the cost, to the dollar.** Priced in money at 0.01 lot and
  adding every cost back: dev pays $111,469 of cost to lose $118,483, leaving a
  gross of **-$7,013** - 6.3% of the cost paid. Validation: $40,462 of cost,
  -$39,638 net, **+$824** gross. Per cell, `net_2R = -0.984 x cost_r - 0.015`
  with **r = -0.995**. A slope of -1 and an intercept of zero is what "no edge,
  pay the cost" looks like.
- **The barriers resolve like a fair coin.** A 2R target against a 1R stop is
  hit first exactly 1/3 of the time under a driftless random walk. Restricting
  to barrier-resolved trades, the realised hit rate climbs from 29.5% at 1m to
  **33.8% at 4h** as the round turn vanishes - landing on the coin's value to
  within half a point, and reproduced on validation (34.0%). The shortfall at
  the fast end is precisely the spread moving the barriers.
- **The engulfment does no work.** A control that keeps the whole geometry and
  drops only the engulfment does *better* (gross -0.0062R against -0.0098R), and
  the setup beats it in only 17 of 44 cells. Forward return from the entry mid
  is negative in 30 of 44 cells, significantly negative in twelve against three
  significantly positive, and 20-bar MFE and MAE are the same size everywhere.

Twenty cells are positive on both splits; all are slow, low-cost cells where a
zero gross edge is no longer swamped by the round turn, none clears t = +2 on
dev, and on validation their profit is **entirely long-side** (+0.487R long
against -0.153R short at 4h, during gold's 2024-25 run). That is drift capture,
available more cheaply from `qlab.strategies.risk_managed_long`.

One methodological note worth carrying forward: pricing these trades "at mid" to
strip costs is **biased upward**, because a long's stop triggers on the bid and
the mid is already past the level - measured at 0.37-0.40 spreads per exit over
300k stop-outs. The unbiased cost-free reads are forward return at a fixed
horizon, and adding measured cost back in money.

The test split was not spent: nothing reached it.

## The engulfing bar's quadrants - **rejected**

A different pattern from the one above, despite the name: the bar must take out
the prior candle's *high* (a wick condition) and close below the prior candle's
*open*. Divide that bar into quarters with `0, 0.25, 0.5, 0.75, 1` anchored at
its low, work a resting sell order in the 0.25-0.5 zone, target the low.
Implemented in `qlab.strategies.engulfing_quadrant`, run tick-by-tick on all
four instruments across **eleven timeframes from 1m to 4h**. Full report:
`docs/findings/engulfing-quadrant.md`.

1,386,322 signals, 784,381 fills, 772,796 trades. The idea names no stop and one
target, so three entry conventions x three stops x five targets were resolved
against the same fills. Pooled over dev and validation the stated geometry loses
**0.502R per trade**, is positive in **1 of 88** cells, and none of the 1,320
(cell x stop x target) combinations clears t = +2. Four findings:

- **The rejection is algebraic, not empirical.** For entry `e`, stop `s` and
  target `t` on the bar's own ladder, the break-even strike rate at zero cost is
  `(s-e)/(s-t)` - which is exactly the driftless first-passage probability. The
  two coincide for *every* triple, so the trade is a fair bet by construction and
  costs can only push it below zero. No zone, stop, target or timeframe can
  rescue it; only a pattern that puts drift into the walk could, and this one
  does not.
- **The eleven-rung ladder is that algebra playing out.** As cost/R falls from
  0.520 to 0.028, the realised strike rate closes on the fair coin (-0.144 to
  -0.006) and never on break-even (-0.550 to -0.040). At 1m and 2m the round turn
  *exceeds the entire payoff* - break-even above 95%, and above 100% at 1m, so
  the trade is unwinnable at any strike rate.
- **The retracement entry selects against the trader.** 36% of signals are
  *missed* - price reaches the bar's low without ever returning to the zone - and
  those bars closed hardest on their lows. A backtest that checks the target
  before the entry books all 354,858 dev misses as wins.
- **Raw comparisons against the nulls are an artifact.** The setup beats both
  size-matched nulls on mean R in **160 of 176** comparisons, which looks
  decisive and is not: signal bars close near their lows, moving the entry closer
  to the target, which raises the fair-coin rate and cuts the payoff together.
  Normalised by each variant's own geometry the advantage vanishes and the setup
  is the worst of the three - notably it lands 0.094 below its own fair coin
  against 0.089 for a null that keeps everything except the stop run, so the
  sweep itself is worth about -0.005. The bullish mirror reproduces the setup to
  0.0007R, and the stricter "prior candle must be bullish" reading is worse
  still.

The test split was not spent.

## Session opening-range breakout - **rejected**

`XAUUSD_SessionBreakout_2026.mq5`, re-implemented tick-by-tick in
`qlab.strategies.session_breakout` and run over the whole corpus on all four
symbols. Full report: `docs/findings/session-breakout.md`, or the
[evaluation page](https://claude.ai/code/artifact/193c264e-6037-4b85-8ec9-7411830b5741).

Results are in **R** - multiples of the bracket width, which is the distance to
the stop - so they are independent of lot size and volatility targeting, and
comparable across instruments and regimes.

| symbol | trades | total R | mean R | dev t | val t | test t | 2026 t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAUUSD | 10,906 | +39 | +0.004 | **-2.15** | 1.95 | 2.06 | 2.43 |
| USDJPY | 10,449 | +258 | +0.025 | 1.95 | 0.10 | -0.99 | **-2.20** |
| USTEC | 9,523 | -26 | -0.003 | -1.00 | 1.43 | 0.21 | 1.07 |
| EURUSD | 9,622 | -490 | -0.051 | -0.80 | **-2.13** | -1.78 | -1.12 |

Four findings, in order of weight:

- **The 2026 re-tune was fitted inside the locked test split.** The expert's
  header tunes "for the 2026 regime" and cites 66 out-of-sample sessions; `test`
  runs 2025-07-01 to 2026-09-01. The split is spent for this strategy family.
- **It loses over dev.** 6,295 gold trades, mean R -0.057, daily t = -2.15, a
  450R drawdown - and -0.053 with dev's own dearer cost profile.
- **What improved is cost, not edge.** Cost per unit risk fell 6.7x, 0.089R to
  0.013R, because the bracket widened $5.51 -> $24.03 while the round turn did
  not. That is real, durable, and scale-free - and it argues for *shorter*
  brackets, not the 60-minute ones the re-tune adopts.
- **The signal predicts nothing measurable.** Fixed-horizon forward return from
  the true entry mid: t = 0.68 on dev and **0.43 in 2026**, the year it was
  tuned on. Only 2024 clears t = 2, and it does not repeat.

A measurement trap worth remembering: taking that forward return from the last
one-minute *bar close* before entry rather than the entry mid manufactures a
spurious +0.13R at t = 11 in every period, because the stale close sits on the
wrong side of a breakout fill. The trade tapes now carry entry and exit mids so
it cannot recur.

The MT5 code itself is sound where it matters - the re-init adoption pass, the
DST-anchored flatten and the stale-position sweep are all correct. Its one
serious live-trading defect is that position size ignores the stop distance, so
risk per trade spans 14x at a fixed lot size.


## The overnight-intraday reversal family - **rejected**

Liu, Liu, Wang, Zhou and Zhu (2025): split the daily return into an overnight
leg and an intraday leg, and the traditional short-horizon reversal turns out to
be carried by one component - sort the cross-section on yesterday's *overnight*
return, hold today's *intraday* return, contrarian ("CO-OC"). Implemented in
`qlab.strategies.overnight_reversal` and run on all four instruments as one
zero-investment cross-section, 976 dev sessions and 366 validation sessions,
under three weighting schemes. Full report:
`docs/findings/overnight-reversal.md`.

These are spot CFDs with no exchange close, so "overnight" is defined the way
the exchanges define it: the session is the regular trading hours of the
matching futures contract (NQ, GC, 6E, 6J) and everything outside it is
overnight. Mean daily return in bps of capital, net of measured cost, rank
weighting, dev:

| variant | net bps | t | Sharpe | turnover |
| --- | ---: | ---: | ---: | ---: |
| CC-CC (traditional reversal) | +6.60 | +1.76 | +0.83 | 2.48 |
| OC-OC | +3.26 | +1.12 | +0.54 | 2.00 |
| OO-OO | +0.56 | +0.16 | +0.08 | 2.47 |
| **CO-OC** (the paper's primary) | **-1.05** | **-0.34** | **-0.18** | 2.00 |

**The paper's central ordering is inverted here.** CO-OC is the weakest of the
four under every weighting, and CC-CC - the strategy it is supposed to explain
away - is the strongest. Four findings:

- **CO-OC is its own null.** Shuffling the signals across instruments within
  each day preserves every marginal, the weighting, the turnover and the cost,
  and destroys only the cross-sectional information. Five of ten placebo seeds
  beat the real strategy. The same placebo beats CC-CC 0 times in 10.
- **The mechanism fails informatively.** The paper conditions on cross-sectional
  overnight *dispersion*. Run through the paper's own two-step procedure,
  dispersion predicts the strategy's **volatility** (t = +2.92) and not its
  **return** (t = -0.17). A specification that only ran the numerator would have
  reported a volatility forecast as a signal. For OC-OC the return coefficient
  is significant with the *wrong* sign (t = -2.19).
- **The cost asymmetry reorders the table and the paper never charges it.**
  CO-OC and OC-OC are flat outside the session and pay a full round turn every
  day; CC-CC and OO-OO hold through and pay only for the change in weight.
  CO-OC turns over 504x notional a year.
- **The whole family is 2020-2021.** Split at the sample midpoint, CC-CC goes
  from +13.90 bps (t = +2.52) to -0.72 bps (t = -0.14) and CO-OC from +6.44 to
  -8.56. This is the third study in this directory to land on the same regime
  boundary by a different route.

Nothing clears a multiplicity correction over the family of four - the best raw
p-value in the study is 0.078. Validation agrees and rescues nothing. The one
row where the paper's ordering survives is **weekly** frequency, where CO-OC
leads at t = +1.07 and the weekend is the only genuinely closed window these
instruments have; that is the piece worth retesting on a wider universe.

Two limits stated in the report rather than buried: N = 4 against the paper's
dozens per asset class, so this can say "not found here" and not "not there";
and VIX, sentiment and the Fama-French factors are outside this corpus, so the
paper's risk-adjustment and GRS pricing tests are **not implemented and not
approximated**. The test split was not spent.

## Gold structural breaks: entry, sizing and exit taken apart - **rejected**

Quan, Zhong and Jiang's H4 XAUUSD swing-break system: rest a stop order at a
confirmed pivot in the H4 closes, stop 3 dollars, target 20 dollars, size at
0.36% risk, and take half the position off at +1R when a one-minute candle turns
against you. Implemented in `qlab.strategies.structural_break` as one fixed core
with four axes moved one at a time, over 6,262 H4 bars and **100.2 million real
ticks** on dev. Full report: `docs/findings/structural-break.md`.

Dev, real ticks, entry and sizing held fixed:

| run | n | return% | Sharpe | win% | payoff | mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V10 half-exit on | 2,048 | -43.3 | -1.08 | 41.2 | 1.24 | -0.073 |
| **B5 half-exit off** | 1,666 | **-5.3** | **-0.01** | 13.8 | **6.18** | **+0.004** |
| V11_FixedR (no candle filter) | 2,081 | -42.8 | -1.11 | 44.1 | 1.10 | -0.071 |

- **The half-exit is the single most destructive component**, and it is the one
  the paper adds on purpose. Paired stationary-block bootstrap on the common
  daily grid: ΔSharpe **-1.070**, 95% CI [-1.652, -0.481], p < 0.0001, survives
  Bonferroni. It lifts the win rate 27.4 points (z = +18.29) and collapses the
  payoff ratio from 6.18 to 1.24, because it fires on 99.9% of winners and 0.1%
  of losers and caps a 6.67R target at about 3.8R. The system's expectancy lives
  entirely in a small number of large winners; the overlay halves exactly those.
  It replicates on validation.
- **The candle filter carries nothing.** ΔSharpe(V10 - FixedR) = +0.024,
  p = 0.94 on real ticks and p = 0.38 on the interpolated tape, so the null
  result is not an artifact of resolution either.
- **The sizing rules are leverage.** No V11 variant moves Sharpe significantly
  (best +0.202, p = 0.24, dies under Bonferroni), and the 2x risk runs confirm
  it: return moves 1.6x, Sharpe moves by under 0.01. The harness prints
  `Sharpe(3.7 r) - Sharpe(r) = 0.00e+00` next to that table so the return column
  cannot be misread as a result.
- **Backtest resolution errs in the direction opposite to the received wisdom
  here.** A minute-bar tape traversed in candle-direction order hits a long's
  stop before its target, and on a 3-dollar stop against a 20-dollar target that
  bites constantly: bar-level backtesting **understates** the no-overlay variant
  by 0.038 R a trade, which is more than the strategy's entire edge.

- **The entry itself is unproven rather than refuted.** Against a 200-path
  random-entry null with identical stops and targets, the break sits at the
  **51st percentile with the overlay on** (p = 0.49 - exactly random) and the
  **78th with it off** (p = 0.23). Better than random, better than an MA(20,50)
  crossover and better than its own literal-reading `retest` variant, and
  significantly better than none of them. Buy and hold gold returned +30.8% at
  Sharpe +0.51 over the same window.

Every cell of every parameter sweep is negative, and performance improves
monotonically as the stop widens from 1.5 to 4.5 dollars - a 3-dollar stop is
small against gold's ~8-dollar H4 ATR, which is why 59% of trades stop out.

The one positive cell is B5 on validation (+30.5%, Sharpe 0.67, mean R +0.094),
and it did **not** get the test split: it is flat on dev over four years and
1,666 trades, so nominating it would be selecting on validation; its profit
factor is 1.09 against a 42% cost drag, so a 40% error in the slippage
assumption erases it; and 2024 into mid-2025 is the strongest trending stretch
for gold in the corpus, which is exactly what a 6.67:1 breakout system flatters
itself on.

## The 5-minute opening range breakout - **does not carry to validation**

Zarattini, Barbon and Aziz (2024), *A Profitable Day Trading Strategy For The
U.S. Equity Market*: take the first 5 minutes of the US session as a range, let
the sign of that candle pick the side and only that side, enter on a stop order
at the range edge, stop out at 10% of the 14-day ATR, hold the rest to the bell.
The version they advertise adds a **Stocks in Play** filter - only names whose
opening-range volume beats its own 14-day average, top 20 by that ratio - and
returns 1,600% over 2016-2023. Implemented in `qlab.strategies.opening_range`,
resolved **on ticks**, on USTEC with the other three instruments as breadth
checks. Full report: `docs/findings/opening-range-breakout.md`.

Ticks rather than bars because the stop is 25.8 index points against an opening
range of 50 and 1-minute bars whose first-hour range routinely exceeds both: the
bar holding the entry usually holds the stop too, and resolving those in the
trade's favour turns a losing strategy into a winning one.

| | dev | validation |
| --- | ---: | ---: |
| mean R, net | **+0.198** | **+0.077** |
| t (Newey-West) | +2.12 | +0.50 |
| bootstrap 95% | [+0.033, +0.372] | [-0.212, +0.380] |
| hit rate | 18.8% | 18.0% |
| winner / loser | +5.59R / -1.05R | +5.14R / -1.03R |

- **The return shape reproduces exactly.** Fewer than one trade in five wins and
  the whole return comes from the 19% that never stop out and run to the bell.
  It is a convexity trade, not a prediction. The edge is present in all four dev
  years and weakest in 2020, so it is not a COVID artefact - and it decays
  monotonically from 2021 onward.
- **The exit surface is a plateau and the paper is standing on it.** Six stop
  widths by seven exits: every profit target is worse than holding to the bell,
  0.5R is negative everywhere, and the paper's pre-specified 0.10/EOD cell lands
  at the maximum of the 42 searched here without having been one of the choices.
- **Most of the edge is geometry, not direction.** Shuffling the opening
  candle's sign while keeping the calendar, the ranges, the ATR and the stop
  still earns +0.133 R on dev, and the real signal at +0.198 R is beaten by 11
  of 60 shuffles (p = 0.20); on validation, 10 of 60 (p = 0.18). The ordering
  `paper > both sides > opposite side` does hold on both splits, and contra goes
  negative on validation - the sign does something, consistently, and not
  distinguishably from zero.
- **Stocks in Play does not transfer, for a reason worth stating.** The sort is
  backwards where the data lives (0.5-1.0x: +0.296 R; 1.0-1.5x: +0.014 R) and
  the paper's high-relative-volume buckets hold 2 and 15 trades. Relative volume
  identifies a name with a fundamental catalyst; an index has none, never
  reaches 3x let alone the paper's 30x, and there is nothing to select. This is
  the wrong corpus for the paper's actual contribution, not a refutation of it.
- **Breadth is the most encouraging table in the study.** The effect is monotone
  in exposure to the US equity open - USTEC +0.198, gold +0.096, EURUSD +0.002,
  USDJPY -0.051 on dev, same ordering on validation. A resolution bug or a
  stop-placement error would show up on all four; this does not.

Validation's t of +0.50 is not a live candidate, so the **test split was not
spent**. The 15-minute range beat 5 minutes on both splits (+0.223 and +0.202 R)
and is the one pre-registerable follow-up.

## The overnight drift - **rejected**

Boyarchenko, Larsen and Whelan, NY Fed Staff Report 917: US equity futures earn
an annualised 3.7% in the single hour between 02:00 and 03:00 New York - the
European cash open - and it is the only hour that survives a multiple-testing
correction. The mechanism is inventory risk: selling into the US close leaves
dealers long, they unwind into the first European liquidity and charge for the
wait, so the drift should be biggest after a close with negative order imbalance
("buy the dip", BtD) and weak after a rally. Implemented in
`qlab.strategies.overnight_drift` on USTEC, with gold and the FX majors as a
falsification set. Full report: `docs/findings/overnight-drift.md`.

The paper's sample ends December 2020 and this corpus starts January 2020, so
this is very nearly a pure out-of-sample test of a published result - run across
the period in which two ETFs launched specifically to harvest it.

Annualised, after spread and commission, against the paper's own post-cost row:

| strategy | dev | validation | paper |
| --- | ---: | ---: | ---: |
| OD (02:00-03:00) | -0.07 | -0.56 | **-0.54** |
| OD+ (01:30-03:30) | +0.27 | +0.40 | **+0.26** |
| BtD (OD+ after a negative close) | +0.47 | +0.60 | +1.10 |
| Rally (OD+ after a positive close) | -0.02 | -0.20 | - |

(Sharpe ratios, no risk-free rate subtracted.)

- **The paper's honest conclusion replicates almost exactly; its headline
  statistic does not.** OD does not survive its own bid-ask spread - the paper
  says so and gets -0.54, this gets -0.07 and -0.56 - and OD+ lands within 0.15
  of the paper's post-cost Sharpe on both splits.
- **The hour is no longer identifiable.** 02:00 ranks 4th of 23 by t-statistic
  on dev and 12th of 23 on validation, and no hour survives Bonferroni or
  Benjamini-Hochberg on either split. The point estimate does replicate (+2.49%
  a year against the paper's +3.7%); four years is simply too short to isolate
  a 3.7% effect, which is what a t of +1.36 on dev means.
- **The asymmetry has the right sign twice and no significance.** BtD minus
  Rally, paired on the same calendar at the same cost: +0.84 bps a day on dev
  and +1.00 on validation - almost the same magnitude across two independent
  periods - with p = 0.37 both times.
- **The conditioning proxy is the weak link.** This feed carries quotes and no
  trades, so relative signed volume is approximated by signing quote revisions.
  That proxy sorts non-monotonically; the **closing-hour return**, which needs
  no proxy, reproduces the paper's shape on dev (biggest-selloff quintile
  +7.49 bps, t = +2.12, with the flat-imbalance middle quintile at zero exactly
  as the mechanism requires) and is flat on validation.
- **The falsification half passes.** Both FX majors are dead at 02:00 on both
  splits, as the mechanism requires. Gold is not - it matches USTEC on dev
  (+2.50% a year, t = +1.67), which is either a dollar-risk-asset extension of
  the same story or evidence that 02:00 is a general European-open effect. This
  study cannot choose.

BtD is the best strategy in the table on both splits, in the same position
relative to OD+ that the paper puts it, and it is still +2% a year at 3.4%
volatility with a Sharpe interval from -0.6 to +1.4 - a money market fund
matched it over the same period without the overnight leveraged index exposure.
Rejected as a strategy, retained as a documented regularity. The test split was
not spent.

## Gold VWAP/EMA regime filter: a rule set with no backtest behind it - **rejected**

Full report: `docs/findings/vwap-ema-gold.md`. Bhatti (2026) specifies six
conditions on a 15-minute XAU/USD candle - right side of the 200 EMA and of the
session VWAP, a pullback that touches the 50 EMA and closes off it, a pin-bar or
engulfing rejection, above-average volume, above-average range - with a 0.5-ATR
stop, a 3R target, and a trailing stop that fires **only on a candle close**
beyond the 50 EMA. Reported: +0.414R expectancy, 45.3% win rate, Sharpe 3.99.

**The paper contains no backtest**, and says so: its Section 6.1 draws the 247
trades from assumed outcome probabilities (0.30 full win, 0.20 partial, 0.08
breakeven, 0.42 loss) "parameterised from the strategy's structural logic". Every
headline number follows arithmetically from those four inputs; no gold price
enters. What it does provide is a completely specified rule set, so what this
study contributes is the measurement the paper does not contain.

Run on the tick tape: **-0.035R per trade on dev** (486 trades, t = -0.63) and
**+0.067R on validation** (170 trades, t = +0.68) - opposite signs, neither
distinguishable from zero, pooled -0.008R with a bootstrap interval of
[-0.097, +0.084]. The cost-free edge at the mid is +0.032R, so this is not a
strategy costs killed. The assumed probability of reaching 3R, 0.30, measures
at 0.093.

Three findings beyond the headline:

- **The exit surface is flat at zero.** 144 cells - six stop widths, six
  targets, trail on/off, session-flat on/off - all between -0.10R and +0.05R.
  The best dev cell reaches t = +1.27 against the t = 3.57 a single cell needs to
  survive searching 144, and validation's best cell is a *different* corner of
  the grid. The paper's chosen cell sits on a plateau at zero.
- **The six conditions are selective but not informative.** They cut 25,008
  tradeable bars to 486, and removing the rejection-candle condition entirely
  doubles the trade count while moving the mean by 0.006R. The one thing they do
  achieve is beating a random entry, which loses 0.07R - so most of what the
  filter buys is getting back to zero.
- **Section 4.3 contradicts Section 3.4.** The VWAP milestone protocol manages
  the trade as price approaches VWAP, but the entry filter requires price to be
  past VWAP already. The touch fires on 79% of dev trades at a mean floating
  -0.44R: it is an adverse event, not a milestone. Recorded per trade, kept out
  of the trade path.

The paper's cost assumption is about ten times too conservative (0.24R against a
measured 0.038R), the news filter is not implementable here and is absent, and
the volume tests use the tick-count proxy. The test split was not spent.

## Decision trees on next-bar direction: priced at their own turnover - **rejected**

Full report: `docs/findings/decision-tree-intraday.md`. Prajwal et al.
(2024) fit a depth-4 Gini tree per name on nine indicators of 1-minute bars,
label each bar by the sign of the next bar's return, shift the signal forward a
bar, and trade it - reporting a test Sharpe of **5.81** against buy-and-hold's
2.36. Their Section IV.C: *"commissions and slippage were not explicitly
incorporated."*

Fitted per instrument on dev (1.3-1.4M bars each) and measured on validation.
**Held-out directional accuracy is within 0.03% of the majority-class base rate
on three of four instruments** - the classifier does not predict direction at
all. Cost-free, nothing is significant on any instrument (best p = 0.052). The
decisive number is the turnover:

| | turns/day | breakeven bps | actual bps | ratio | net total |
| --- | ---: | ---: | ---: | ---: | ---: |
| XAUUSD | 133.6 | 0.055 | 0.682 | 12.5x | -85.9% |
| USTEC | 113.0 | 0.056 | 0.980 | 17.4x | -91.3% |
| EURUSD | 26.6 | 0.080 | 0.579 | 7.3x | -26.8% |
| USDJPY | 126.4 | 0.037 | 0.825 | 22.2x | -90.3% |

The gross edge can afford a twentieth of a basis point per round turn; the round
turn costs most of one. Depth changes nothing (accuracy varies by 0.003% across
depths 3-6) except to raise turnover monotonically.

The result that explains the rest: the paper's cost-free equity curve is **three
times better when the position is applied to the bar the model does not
predict** (their forward shift, Sharpe 1.02) than to the bar it does (Sharpe
0.30). A real directional edge would degrade under delay. What is actually being
measured is an exposure artefact - a rule in the market 34% of the time has
lower volatility than buy-and-hold and therefore a flattering Sharpe, which is
the shape of the paper's whole Table 1. The matched control is a benchmark
scaled to the same average exposure. The test split was not spent.

## Small-cap retail strategies: an edge exactly the size of its own spread - **rejected**

Full report: `docs/findings/smallcap-poudel.md`. Poudel (2025) specifies six
strategy families for US small caps ($300M-$2B market cap) and reports
out-of-sample Sharpe ratios of **0.82** (volatility-scaled momentum) and **0.88**
(regime-filtered composite) over 2022-2024, against 0.45 for IWM, net of
$0.02-$0.05 spreads and $0.01 of slippage a side.

Families A, D and F rebuilt on a **point-in-time S&P SmallCap 600 panel**
reconstructed from the index change history - 868 tickers, 1.74M ticker-days,
2018-2026 - with the paper's own costs. Three families need minute bars or an
earnings calendar and are out of reach; 211 of 1,079 tickers (19.6%) no longer
resolve, so the panel *flatters* the strategy.

The entry signal is real: family A's cost-free out-of-sample Sharpe is **+1.19**
at +33.6% a year, better than the paper claims net. The problem is what it costs
to collect:

| family, split | median entry | round turn | hold | gross $/trade | cost $/trade | cost as share of gross |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, in-sample | $18.61 | 37.6 bps | 3d | +$4.06 | $4.60 | **114%** |
| A, out-of-sample | $21.13 | 33.1 bps | 3d | +$7.28 | $7.15 | **98%** |
| F, out-of-sample | $21.93 | 31.9 bps | 4d | -$1.78 | $3.27 | - |

The best case in the study clears its own transaction costs by **thirteen cents a
trade**. **All 120 exit configurations swept in-sample - target 0.5-3.0 ATR, stop
1.0-3.0 ATR, horizon 5/10/20 days, both families - are net negative**, so the
verdict does not rest on the exit choice. Out-of-sample family A is +0.09
(t = 0.16, p = 0.88) and family F, the paper's headline, is significantly *worse*
than holding IWM (-0.84 Sharpe, p = 0.049).

Two independent problems with the paper itself: its Section 5.1.3 sizing formula
has no account term at all, so ten positions come to **9-18% gross exposure** and
cannot produce a 12% annual return whatever the edge; and its results table
reports IWM returning **+18.1%** cumulatively over 2022-2024 when IWM actually
returned **+2.2%**, with a family-F year list (-15.1%, +21.4%, +9.8%) that
compounds to +13.2% rather than the +45.2% stated two lines later. The
post-publication period 2025-2026 was **not** spent.

## Machine-learned FX signals: 792 runs, none beat holding the thing - **rejected**

Full report: `docs/findings/fx-ml-enkhbayar.md`. Enkhbayar and Slepaczuk
(2024) run eight regressors (Ridge, KNN, RF, XGBoost, GBDT, ANN, LSTM, GRU) over
technical indicators to predict the next bar's return, threshold the prediction
against training-set quartiles, and hold until the signal changes - all inside a
rolling walk-forward, against an EMA-cross trend follower and buy-and-hold.

Replicated on this corpus - EURUSD, USDJPY, XAUUSD, USTEC, daily and 4-hour,
2022-2025, a period the authors never saw. **792 runs.** Median annualised Sharpe
**-0.21** against **+0.97** for holding the instrument. Paired against
buy-and-hold on the same bars by stationary-block bootstrap:

| | all 792 runs |
| --- | ---: |
| median Sharpe difference vs buy-and-hold | -0.99 |
| positive point estimate | 78 (9.8%) |
| significantly different, raw p < 0.05 | 274, **0 better** |
| significantly different after Benjamini-Hochberg | 156, **0 better** |

The paper's one positive claim - "at least one model beats the benchmark in each
case" - reproduces in 5 of 8 cases and is a max-of-24 artefact: the best paired
p-value anywhere is 0.27. Two specification choices are sufficient causes and
both are measured. The gate compares *predicted* returns to quartiles of
*realised* ones, but the tree and linear models predict with 0.14-0.20 of the
target's spread, so under the paper's own rule the median run is in the market
17-19% of the time and KNN on EURUSD daily never trades at all; the neural
networks fail the other way, predicting 1.5-3.1x wider than anything that
happens. And the features are price *levels*, which a tree cannot extrapolate
past. Fixing both changes which model looks best and still produces nothing that
beats buy-and-hold.

The paper's RQ1 (intraday beats daily) inverts here, with the mechanism visible:
median cost drag is 0.12 Sharpe daily against **0.70** at 4-hour, on 245 trades a
year instead of 37. Its RQ2 (only-sell is best) also inverts - only-sell is the
worst mode here and only-buy the best - which identifies both results as the
sample's drift arriving through whichever one-sided mode faces it, not a property
of the models. The locked test split was not touched.

## Scheduled flows: macro releases and the FX fixings - **rejected on the held-out split**

Full report: `docs/findings/scheduled-flows.md`. Eight hypotheses in three
pre-registered waves, each a clock rule on a calendar known in advance, each
against the same rule on the same clock minutes on days with no event. The
search went here because scheduled events with a named counterparty are the one
place both of this corpus's constraints - minute edges too small, daily samples
too short - can be escaped at once.

**One candidate reached the test split, and it failed there.** The unwind after
the 09:55 JST Tokyo fix on gotobi days - the 5th, 10th, 15th, 20th, 25th and
month-end, when Japanese importers buy USD at the fix (Ito & Yamada 2017):

| short USDJPY 09:55 -> 11:00 JST, gotobi days | n | net bps | t |
| --- | ---: | ---: | ---: |
| dev | 281 | +2.35 | +3.15 |
| validation | 107 | +4.16 | +3.17 |
| **test, run once** | **83** | **-0.51** | **-0.44** |

Before the test it had cleared everything: a pre-registered cell passing
Bonferroni across its wave, a weekday-matched permutation p of 0.0001, adjacent-
day placebos at zero, every year 2020 to H1 2025 positive, positive at every exit
from 10:15 to 15:00, +2.45 bps under doubled spread, fully adverse slippage and
1000 ms latency at once, and +2.94 bps at t = +4.49 on tick-accurate fills with a
one-second entry latency. It was specific to the instrument the mechanism names
- EURUSD, gold and USTEC showed nothing. Against the pre-registered rule the
test is a clean FAIL, about 2.6 standard errors below dev and validation. Read
after the verdict: the everyday post-fix unwind on non-gotobi days went too
(+1.04 bps at the mid before, -0.38 in test), so something changed at the fix
around mid-2025.

Everything else failed earlier:

- **Pre-FOMC drift** on USTEC is +33 bps a meeting, the same in both splits
  (+32.6 / +34.6) and at the published magnitude - on 43 meetings, t = +1.42. The
  one regularity here this corpus can neither establish nor refute.
- **Fading USTEC's first reaction to CPI and NFP** looked like reversal in all six
  cells (NFP 120 m: -28 bps followed, t = -2.52) and then went the **wrong way** on
  347 fresh 08:30 releases (-7.09 bps faded, t = -1.71), with no dose-response
  even in sample.
- **The FOMC statement fade** was positive on all four instruments and does not
  transfer to the minutes, the same 14:00 minute without a press conference.
- **The WM/R 16:00 London fix fade** loses on every instrument; month-end hedge
  rebalancing is underpowered and unsigned.
- **Two further benchmarks, added afterwards** - the Shanghai Gold Benchmark
  auctions and the ECB reference rate, the same fade shape - fail all three
  primary cells (t -2.42, -1.64, -1.12). Every mid stays within 1 bp of zero and
  of its control hour; the losses are the round turn.

The first clean out-of-sample failure of a candidate that passed every in-sample
check this project has: the case for the locked split, made by the only
candidate that got far enough to use it.

## Round numbers on EURUSD, USDJPY and USTEC - **rejected**

Full report: `docs/findings/osler-round-numbers.md`. Osler's order-book
mechanism - take-profits at round numbers, stops just beyond - was tested only
on gold, and Osler's evidence is FX. The same `qlab.levels` harness, unchanged,
on the three untested instruments with nested grids (EURUSD 10/50/100 pips,
USDJPY 0.10/0.50/1.00, USTEC 50/100/500 points).

**All nine pre-registered primary cells fail** - coarsest grid x touch, sweep,
breach x three instruments, per-day t at 30 minutes against a Bonferroni 2.77.
The best is EURUSD big-figure sweeps at +0.70, negative on validation. Every
real mid sits inside its 12-draw date-shifted placebo.

- The fine grids are strongly negative per day (t down to -17) with a mid near
  zero: several events a day, a round turn each, no edge.
- The one ordering in Osler's favour is EURUSD sweep reversion growing with the
  grid (+0.35 / +0.39 / +1.56 bps at the mid, 120 minutes) - the right shape, on
  the paper's own instrument, inside its placebo band.
- USDJPY big-figure sweeps *continue* (dev per-day t -2.74), backwards exactly as
  gold's were. Two instruments now have the reversion half the wrong way round
  and none has it the right way.

The test split was not spent.

## Rebalancing flow into the US close - **rejected**

Full report: `docs/findings/close-rebalance.md`. Leveraged index funds and
short-gamma dealers trade with the day's move near the close, so the last half
hour should continue it (Cheng & Madhavan 2009; Baltussen et al. 2021). The
pre-registered cell - trade the sign of prior-close -> 15:30 ET, hold to 16:00,
USTEC - is **-0.13 bps at t = -0.11**, and validation goes significantly the wrong
way (-4.28 bps, t = -2.64). The dose-response slope is +0.0175 at t = +1.03. Gold,
a falsification instrument, is the only one with a significant slope, and was
not pursued for exactly that reason. The test split was not spent.

## The 15-minute opening range, pre-registered - **inconclusive on test, not deployable**

Full report: `docs/findings/orb15.md`. The follow-up the 5-minute report named
as its one legitimate single shot. Dev +0.223 R (t = +2.20), validation +0.202 R
(t = +1.14), pooled +0.217 R at t = +2.46 - and again the direction does no
measurable work: a shuffled-sign placebo earns +0.178 / +0.078 R and taking both
sides +0.171 / +0.231. It went to test as a geometry edge with a stated
low-power warning and returned **+0.086 R at t = +0.57** over 256 sessions -
+0.262 R in H2 2025, -0.042 in 2026. Inconclusive under its pre-registered rule;
the series decays year on year. The test split is spent for the opening-range
family.

## Discovery program: tail-conditioned structure - **rejected, one observation**

Full log: `docs/findings/discovery-program.md`. Scripts:
`scripts/research/discovery_batch1.py`, `scripts/research/discovery_batch2.py`.
A running, pre-registered search rather than a single study: hypotheses carry
IDs and parents, each batch is written into the log before its script runs.

The premise: minute-scale predictability here is real and 0.2-0.5x a round
turn, and every earlier test of it was linear. If the response to a move grows
with the move, conditioning on the tails raises the edge while the cost stays
fixed. **Batch 1 - 24 pre-registered cells on dev, Sidak |t| 3.07 - found none:**

- After a 4-sigma 15-minute move the next hour is a random walk: max-favourable
  and max-adverse excursions are equal on all four instruments, with no
  dose-response in the threshold.
- Scheduled against unscheduled jumps makes no difference.
- A market that under-reacts to a shock elsewhere does not catch up.
- A compressed range predicts expansion (forward range 1.45-1.62x vs 1.0x) but
  not which way.

**Batch 2** took the one coherent thread - USDJPY continues where the other three
do not, in three separate constructions - to validation as a single fixed cell.
Dev: 27 of 27 grid cells positive, dose-response in move size, not drift
(+2.45 bps hour-and-year-adjusted against +2.42 raw). Validation: **+1.07 bps at
the mid, +0.25 net, t +0.58** - a fail under the rule. The rule was underpowered
(expected t 1.38 at the dev effect size), which is recorded as a registration
error; pooled over both splits the net is **+1.31 bps, 95% CI [-0.33, +2.95]**.
An observation, not an edge. The test split was not spent.

**Batch 3** conditioned on the two other things the feed carries - the spread
and the tick count - and on the weekend. Zero of nine primary cells pass. Tick
activity sorts nothing; the weekend reverses on USTEC and gold and continues on
FX, which pools to noise. Spread shocks were registered as a reversal and came
out backwards: at 120 minutes the displacement made while the spread was blown
out **continues** on all four instruments (pooled +5.30 bps, t +2.67 on dev) -
the Glosten-Milgrom signature of informed flow. It was not taken to validation:
the expected t there is 1.26, because the broker rebuilt its spread
distribution in Q4 2023 and the trigger almost stopped firing everywhere except
USDJPY.

The program's conclusion is about power. The effects that recur are 1-9 bps a
trade against 28-70 bps of noise, which needs about 1,300 independent trades -
seven to eight years - to confirm at t 2.5. The held-out data here is 2.7 years.

**Batch 4** (`scripts/research/discovery_batch4.py`) took the one kind of test
the power arithmetic still allows - a strategy trading many times a day on all
four instruments, which is the only kind validation's 390 days can confirm (it
needs a net Sharpe near 2). A gradient-boosted forecast over 24 state variables,
one model per instrument, fitted on 2020-2022 and read on 2023 inside dev, traded
only when the forecast cleared cost. **Both horizons fail their gate**: at 30
minutes the pooled net is -0.72 bps a trade (per-day t -2.84), at 120 minutes
-0.52 (t -2.06). Out-of-sample correlation of forecast with outcome is -0.004 to
+0.024, and randomising the direction does about as well. The forecastable mean
at the mid is about +0.3 bps - below this account's commission alone (0.2-0.45
bps), so no execution style closes the gap.

**Batches 5-7** (`scripts/research/discovery_batch5.py` .. `discovery_batch7.py`)
ran the conditional grid the program had not: liquidity sweeps by level, depth
and reclaim time, breakouts that hold, extreme-move fades, conditional
autocorrelation, opening drives and USD dislocations, crossed with session,
volatility regime, USD direction and intraday trend, at 1-60 minutes - **81,040
cells**, screened on odd months of dev under Benjamini-Hochberg, confirmed on
even months, and re-run on placebo outcomes. Net of today's cost, zero cells pass.
At the mid, **432 cells replicate across the halves against 14-22 on placebo** -
real reversal structure, mostly smaller than the round turn. The 23 cells that
clear cost, pooled and filled on ticks at 250 ms, earned +0.44 bps a trade on the
confirmation half (t 3.85) and earned a validation shot, where they lost **-0.44
bps a trade at t -3.57**: the mid fell from +1.17 to +0.09. The structure belongs
to the 2020-2023 quoting regime, and interleaved halves - which share an era by
design - could confirm it replicated but not that it persisted. Sixteen
hypotheses, none validated; the next step is an owner decision between searching
the post-2023 regime (spending test), forward time, or new data.

## Liquidity sweeps and order blocks (UAlgo) - **rejected**

Two components of UAlgo's *Price Action Toolkit Lite* TradingView indicator,
replayed bar for bar in `qlab.strategies.sweep_orderblock` and traded tick by
tick on all four instruments at **twelve timeframes, 1m to D1**. Full report:
`docs/findings/sweep-orderblock.md`, pre-registered before any outcome was
computed.

* **Liquidity sweep** - a 30/30 pivot level taken by a bar that closes back
  inside it; enter against it at market, stop at the sweep bar's wick.
* **Order block** - after a close through the latest zigzag swing (BoS/CHoCH),
  the extreme of the bars since that swing, one ATR deep; rest a limit at the
  near edge while the block is among the newest two shown, stop at the far edge.

The indicator trades nothing, so both stops were run with and without an ATR
buffer, against 1R/2R/3R targets and no target: 8 exits per setup, 768
combinations on dev. Pooled over dev and validation at the stated 2R exit:

| | trades | mean R | win | fair coin | cost/R |
| --- | ---: | ---: | ---: | ---: | ---: |
| liquidity sweep | 126,787 | **-0.471** | 0.289 | 0.333 | 0.355 |
| order block | 358,408 | **-0.528** | 0.275 | 0.333 | 0.285 |

**0 of 96 primary cells pass on dev**, and 3 of 768 exit combinations clear
t = +2 against ~19 expected by chance; 0 of 751 on validation. Four findings:

* **Both converge on the coin from below as the timeframe slows and never cross
  it.** The loss tracks cost/R down the ladder, the shape of every price-action
  study here. The one positive pooled rung, sweeps at 4h (+0.137R on 172 dev
  trades), is -0.365R on validation.
* **The structure break carries nothing.** Normalised for cost, a block built at
  a break of structure and the same zone built at a random bar with no break
  differ by 0.004 in strike rate on dev; CHoCH and BoS blocks lose the same to
  0.001R.
* **Block fills are adversely selected.** After a limit fills at the block,
  price keeps going through it - -0.198 bps over the next minute at t -11.4 on
  dev, twice a random zone's. The moves that bounce short of the near edge never
  fill.
* **The sweep's reversal is a 2020-2023 property.** On dev a 1m sweep reverses by
  0.06 bps over the next bar (t 2.1), a tenth of a round turn; on validation the
  swept side *continues* (-0.147 bps at five bars, t -2.5). It beat its null by
  0.044 in cost-normalised strike rate on dev and lost to it on validation - the
  regime boundary `discovery-program.md` found by another route.

**Addendum A - to the opposite block, also rejected.** The owner's reading of
the trade: enter at the start of the block, stop at its other end, take profit
at the opposite-side block the chart shows. Pre-registered as 48 cells at the
opposite block's near edge; **0 pass on dev**. Pooled over both splits:
315,565 trades, **-0.545R**, strike rate 0.143 against a per-trade fair coin
of 0.191, 4 of 95 cells positive, -$38,540 at 0.01 lot - no better than the
fixed 2R target (-0.528R) and below the null zones' margin. Opposite blocks are
reached *less* often than a random level the same distance away, so they are
not the magnet the reading assumes; targeting their far edge changes nothing
(-0.544R).

**Addendum B - the opposite trade, also rejected.** Taking the other side of
every trade loses about the same money: at 0.01 lot over both splits the
sweep goes from -$13,927 to -$12,280 mirrored (original target as stop,
original stop as target) and -$9,715 reversed; the block from -$40,645 to
-$42,107 and -$36,675. 0 of 96 primary cells pass for either reading. **A
trade and its exact opposite both win less often than their fair coin**, which
a directional edge cannot do - the shortfall is the spread, slippage and the
stop-first tie, not a direction. The opposite of a resting limit is also not
the same trade reversed: a stop at the block's edge triggers half a spread
sooner, on touches the limit never filled, and pays slippage. The one rung
positive in both splits, order blocks at 4h (+0.055R mirrored, t 2.1 over
645 trades), was selected after looking and is too small for the test split
to confirm (expected t about 1.0).

The test split was not spent.


---

## Precision Sniper confluence scoring (WillyAlgoTrader) - **rejected**

A TradingView confluence indicator (Pine v6 v1.4.0): ten weighted factors
scored per bar, an A+/A/B grade, an EMA-cross trigger gated on the score, and
a structure stop with three R-multiple targets on a trailing ladder. Ported in
`qlab.strategies.precision_sniper` and traded tick by tick on all four
instruments at **twelve timeframes, 1m to D1**, across **all six parameter
presets** and both settings of its HTF filter. Full report:
`docs/findings/precision-sniper.md`, pre-registered before any outcome.

17,780,538 signals resolved - 552 cells, 44 exits each (4 stop constructions x
11 targets, including the script's own managed ladder simulated on the tape),
against a size-matched mechanics-only null. Pooled dev + validation at the
script's own configuration: **-0.2535R** over 6.02M trades, against a null of
**-0.2586R**. **0 of 24,244 grid combinations pass on dev**; 2 of 24,102 on
validation, and those two are the same five daily trades counted twice.

| | dev | validation |
| --- | ---: | ---: |
| mean R at zero cost, setup | -0.029 | +0.011 |
| mean R at zero cost, matched null | -0.026 | -0.009 |

Four findings:

* **Mean R is a readout of cost/R, not of skill.** It climbs monotonically
  from -0.347 at 1m (cost/R 0.250) to +0.102 at D1 (cost/R 0.017) and crosses
  zero where the round turn stops taking ~2% of the stop - while the TP3
  strike rate stays between 0.099 and 0.145 at *every* timeframe on both
  splits. Fitted across all 552 cells, the zero-cost intercept is
  indistinguishable from the null's and changes sign between splits.
* **The signal carries no direction.** Forward mid return from the fill, no
  barrier and no cost: across 24 timeframe-by-split cells the largest |t| is
  3.16, and it is negative (-0.058 bps, dev 1m). There is no horizon at which
  these entries predict the next move.
* **Three of the four filters cannot bind.** Trade counts are identical at
  minimum scores 0, 3, 4 and 5, because the trigger conditions themselves
  award 1.5 of the 8 available points to every signal; hide-C removes 0.6%;
  and at the HTF filter's **default empty setting the 1.5-point HTF factor -
  the heaviest of the eight - is awarded to 0.000 of signals**, because
  `request.security("", ema[1])` reads the chart's previous bar, which on a
  crossover bar is by construction the pre-cross state. Switching a real HTF
  on moves the A+ share from 18% to 46% without moving the P&L.
* **`Auto` selects the worst cells in the study.** It maps 5m and below to
  `Scalping`, where cost/R peaks: EURUSD 1m is -0.486R at t -57.7 on dev and
  -0.566R at t -33.5 on validation.

Two of the ten factors - the volume surge and VWAP - are not computable on a
quote feed with no size, so the script's own adaptive scale ran the engine at
max 8.0; the volume factor is in any case added to the bull and bear scores
identically and cannot discriminate direction. The test split was not spent.

---

## The New York open EMA rule

Full report: `docs/findings/open-ema.md`. Strategy in
`qlab.strategies.open_ema`; run by `scripts/backtests/backtest_open_ema.py`.

A creator's rule for NAS100 on 5 minutes: take the 09:30-09:35 candle, go long
if it closes above the 12-period EMA and short if below, then trail the stop and
"stay in as long as momentum continues" at 1% risk. Reported over 2019-2026:
1,448 trades, +982%, 57% wins, PF 1.29, 19.7% max drawdown. Tested on USTEC over
960 dev and 368 validation sessions, tick-resolved, with the unspecified exit
swept across three families at seven distances.

**Rejected.** Four findings, in the order they bind:

* **The rule collapses before it is tested.** `c > ema(c)` rearranges to
  `c > ema(c).shift(1)` for every span, so the 12 does no work in choosing the
  side - it survives only in how far the close sits from the level. The rule is
  a one-bit momentum print: did the first five minutes close above where the
  previous hour settled? The upside is that the specification's only real
  ambiguity - whether the signal candle is in its own average - dissolves.
* **The stated exit is the weakest of the three families.** Trailing tops out at
  +0.052 R per trade pooled; a *fixed* stop with no trail reaches +0.224 R, four
  times as much. Tight trails are stopped on 100% of sessions with a 2-8 minute
  mean hold and lose money net of cost - `trail @ 0.05` has the highest gross
  edge in the study (+0.152 R, t +3.93 on dev) and is -0.014 R after a 1.9-point
  round turn. **No cell in the 21-cell grid clears q = 0.10.**
* **It does not beat a coin flip on its own fills.** As specified, +0.048 R
  pooled (t +1.65, bootstrap p 0.084) against the coin's +0.009 R, difference
  p = 0.286 - and on validation the coin won outright. The cell that *does*
  reproduce the headline return (+942% at 1% risk, PF 1.26) is the fixed-stop
  cell, not the rule, and it too fails against the coin (p 0.170, and the coin
  beat it on validation +0.254 R to +0.222 R). Compounding at fixed fractional
  risk turns a per-trade difference no test can resolve into an eight-fold gap
  in total return, which is why the report reads the edge and the headline apart.
* **The reported statistics are internally inconsistent with a trailing stop.**
  57% wins at PF 1.29 implies a payoff of 0.97 - winners the same size as losers
  - and a trail makes the opposite shape. Across the sweep the win rate climbs
  from 35.5% to 52.3% only as the payoff falls from 1.62 to 1.02, and **0 of 21
  cells reach both numbers.** That is a claim about the reported figures rather
  than about this corpus, and it points at a profit target rather than a trail.

The side does have the right sign in both splits (+0.023 and +0.042 ATR to the
bell) and the edge orders monotonically in |close - EMA|, which is what the
mechanism predicts - but the coin control orders the same way, so most of that
gradient is the volatility of a wide opening candle rather than information.
The whole (small) effect also depends on the EMA including overnight tape: a
regular-session-only EMA has a signal three times larger by magnitude and no
edge at any distance. The test split was not spent.
