# Two SSRN strategies, on the real tape - evaluation

Two papers, tested end to end on this corpus. They are not the same kind of
claim and they do not get the same verdict.

| | Paper A | Paper B |
| --- | --- | --- |
| | Chen (2025), SSRN **5246516** | Tan, Zhang & Zhu (2026), SSRN **5757622** |
| | *AI in Day Trading: an Intraday Trading Framework with Economic Indicators and LLM Analysis* | *Forecasting Intraday Trading Volume with Periodicity* |
| Claims | 730% total / 33.5% a year / Sharpe 1.67 on USTEC, 2018-2025; an LLM sentiment filter lifts it to 983% and Sharpe 2.11 | out-of-sample R² 0.293 vs 0.103 for the same features untransformed; VWAP tracking error 7.013 bp vs 7.275 (OLS) and 8.804 (equal weight) |
| Kind | a directional alpha strategy | an execution / forecasting model |
| **Verdict** | **Not deployable.** No measurable edge. | **Half deployable.** The forecast is real; the money is not. |

**Paper A.** Implemented exactly and resolved on ticks. Over the full 5.4-year
non-test sample, **212 trades, +0.02 Sharpe, t = +0.04, p = 0.97, −$1.20 per
trade** on $10,000. Dev (2020-2023) loses 22.6%; validation (2024 - H1 2025)
makes 20.1%; the two disagree in sign and neither is individually significant.
The edge is negative *at mid*, before any cost, on the larger split - so this is
a signal failure, not an execution one. The held-out test split was not spent.

**Paper B.** The forecasting claim replicates emphatically: the
within-transformation beats identical features without it in **64 of 64**
head-to-head comparisons across four instruments, two intraday grids and two
splits. The execution claim does not: across 16 paired bootstraps, SPAR³ tracks
VWAP significantly closer than OLS³ in **1**, significantly *worse* in **4**,
and indistinguishably in **11**.

The two verdicts have the same shape for opposite reasons, and the pairing is
the useful part of this study: Paper A is a strong claim about profit resting on
a mechanism that turns out not to exist; Paper B is a strong claim about
accuracy that is entirely true and still does not turn into profit.

---

## 0. What this corpus can and cannot say

**Paper A: the backbone transfers; the LLM layer cannot be tested here.** Rules
1-5 are pure price mechanics and are implemented exactly. The LLM overlay needs
the *actual, forecast and previous* values of each release. This corpus is four
price feeds; it holds none of them. Nothing below claims to have replicated the
LLM version, and §5 says what can be said about it on the paper's own evidence.

**Paper A: the calendar is reconstructed.** There is no economic calendar in
this project. Every release date in `src/qlab/macro_calendar.py` is generated
from a publication rule a trader knew in advance - never from the tape, which
would be circular for a strategy that trades post-release volatility. §1 audits
the reconstruction before a single trade is taken.

**Paper A: the sample is shorter and later.** 2020-01-29 onward, so the
2018-2019 third of the paper's window is missing.

**Paper B: volume is quote count, not share count.** These are OTC CFD feeds
with no consolidated tape and no size on a quote. `n_ticks` - quote revisions
per bar - is the stand-in, and it is the one an FX/CFD execution desk actually
works with. Everything measured is scale-free (R² is a variance ratio, VWAP
weights sum to one), so the units cancel where it matters.

**Paper B: the CMEM-Kalman benchmark is not reimplemented.** It is an
EM-estimated state-space model and a project of its own. The comparison
carrying the paper's actual argument - within-transformed against not, on
identical features - is exact, as is the historical-mean benchmark that defines
out-of-sample R².

---

# PAPER A - the macro-news breakout

## 1. Is the reconstructed calendar real?

Median 1-minute range in each release minute, against the same clock minute on
every day the rule did *not* name. Measured over dev + validation only: the
audit informs no parameter, but a locked split is locked for looking at as well
as for trading.

| event | tier | n | release range | control | ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Nonfarm Payrolls | exact | 61 | 73.80 | 6.23 | **11.84×** |
| CPI | approx | 64 | 67.92 | 6.23 | **10.90×** |
| FOMC Statement | exact | 44 | 57.98 | 8.85 | **6.55×** |
| GDP | approx | 61 | 26.12 | 6.23 | 4.19× |
| Initial Jobless Claims | exact | 273 | 18.91 | 6.23 | 3.03× |
| PPI | approx | 64 | 16.73 | 6.23 | 2.68× |
| Retail Sales | approx | 64 | 12.31 | 6.23 | 1.98× |
| ISM Manufacturing | exact | 64 | 30.61 | 17.30 | 1.77× |
| ISM Services | exact | 64 | 27.69 | 17.30 | 1.60× |

Every rule lands. The NFP rule is mechanical and exact - reference week
containing the 12th, third Friday after it closes - and reproduces every
published date checked against it (`tests/test_macro_calendar.py`). FOMC dates
are the Fed's own published schedules.

This table also *caught a bug*. The first CPI rule ("nearest weekday to the
12th") scored **1.39×** while PPI, defined as the business day after it, scored
**4.54×** - the rule was systematically one day early, and what the code called
PPI was landing on the actual CPI release. Restricting CPI to the Tue/Wed/Thu
that BLS actually uses moved CPI to 10.90× and PPI down to its own honest
2.68×. That correction came from a publication regularity - BLS never released
CPI on a Monday or a Friday in this period - not from fitting the tape.

**Caveat that stays.** The four `approx` events are right about the week and
right about the day maybe half the time. Results are therefore always reported
for the exact tier separately.

## 2. What the Trading Indicator actually is

The paper specifies a Kalman filter with `Q = 0.01`, `R = 100`, wrapped in a
sigmoid. Substituting those constants, the Kalman gain converges to
**K = 0.00995** - a constant. A constant-gain filter on a random walk is an
exponential moving average, so `p_hat` is a **~100-minute EMA** and the filter
contributes no state estimation beyond it.

The thresholds then unpack to a plain distance from that EMA:

| TI | implied deviation | at USTEC 15,000 |
| ---: | ---: | ---: |
| 20 | −0.401% | −60.1 pts |
| 30 | −0.244% | −36.7 pts |
| 50 | 0.000% | 0.0 pts |
| 70 | +0.244% | +36.7 pts |
| 80 | +0.401% | +60.1 pts |

So the rule is: *two minutes after a release, if price is more than a quarter of
a percent above its 100-minute EMA, buy the 200-minute high.* That is a
defensible momentum filter. It is not a Kalman filter doing any work, and a
reader who takes "integrates a Kalman filter with a sigmoid function" at face
value will overestimate how much machinery is in here.

**Table 1 also contradicts the body.** It lists thresholds as "30/70", which
read literally means buy above 30 and sell below 70 - overlapping conditions
that fire on nearly every event. The body describes a *buffer zone between* the
thresholds, which only exists under buy > 70 / sell < 30. §5 runs both.

## 3. The rule as written, tick-resolved

Entry and exit are both touch-triggered, so fills come off the quote tape, not
off bar highs and lows. Costs are the measured spread, the contract commission,
and measured latency drift.

| | dev (2020-2023) | validation (2024 - H1 2025) | **pooled** |
| --- | ---: | ---: | ---: |
| events → signals → trades | 554 → 217 → 163 | 211 → 70 → 49 | 765 → 287 → 212 |
| net return | **−22.6%** | **+20.1%** | −2.6% |
| CAGR | −6.2% | +13.1% | ≈0% |
| Sharpe (daily) | −0.57 | +1.33 | **+0.02** |
| Lo *t*-stat on Sharpe | −1.24 (p = 0.22) | +1.79 (p = 0.073) | **+0.04 (p = 0.97)** |
| max drawdown | −27.1% | −4.9% | — |
| win rate | 39.9% | 61.2% | — |
| profit factor | 0.78 | 1.87 | — |
| net per trade | −$13.88 (p = 0.09) | +$40.98 (p = 0.04) | **−$1.20 (p = 0.85)** |
| **edge at mid**, no costs | **−$11.68** | +$43.29 | +$1.02 (p = 0.91) |
| all-in cost | $2.20/trade | $2.31/trade | — |

Exact-date tier only: dev −21.1% (−$15.64/trade, p = 0.12), validation +10.0%
(+$24.33/trade, p = 0.22). Same story, smaller, and the validation result stops
being nominally significant once the guessed dates are removed.

*p*-values are stationary block bootstraps and move by a point or two between
draws; they are quoted to two decimals for that reason.

Three things to read off this table.

**The pooled result is zero.** Not "small", not "marginal" - 212 trades over
5.4 years produce a Sharpe of 0.017 and a *t*-statistic of 0.04. The paper's
1.67 is not in this data at any magnification.

**On the losing split it loses at mid.** −$11.68 per trade before spread,
commission or slippage, against an all-in cost of $2.20. Costs are 16% of the
damage. This is a signal that does not work, not an edge eaten by friction -
which matters, because the two failures call for opposite responses and only one
of them can be fixed by a better broker.

**The two splits disagree in sign,** and by year the sign flips repeatedly:

| year | trades | net/trade |
| ---: | ---: | ---: |
| 2020 | 33 | −$31.9 |
| 2021 | 29 | +$10.7 |
| 2022 | 58 | −$21.8 |
| 2023 | 43 | −$5.8 |
| 2024 | 35 | +$29.4 |
| 2025 (H1) | 14 | +$70.0 |

The profitable years are the two trending years; the losing years include the
2022 bear market and the 2020 crash. That is what a long-biased momentum rule
with no edge looks like when the market itself decides the outcome. The
regime story is post-hoc and offered as description, not defence.

## 4. The control: is the *news* doing anything?

Same mechanics, same clock minutes, same number of events - moved to days with
no scheduled release. If the news matters, the calendar should beat this.

| | calendar | control | news premium |
| --- | ---: | ---: | ---: |
| dev | −$13.88/trade (p = 0.09) | −$10.51/trade (p = 0.52) | **−$3.38** |
| validation | +$40.98/trade (p = 0.04) | −$8.01/trade (p = 0.61) | **+$48.99** |
| pooled | −$1.20 (p = 0.85) | −$9.90 (p = 0.40) | +$8.70 |

On the larger split the news is *worse than nothing*. On the smaller one it is
worth $49 a trade. A premium that changes by $52 between splits, on 49 trades,
is not a premium - it is the same instability §3 already showed, seen from a
second angle. The control's own pooled p of 0.40 says the mechanics stripped of
news are also worth nothing, which at least is consistent.

## 5. The threshold reading, and the LLM layer

The threshold ambiguity is not the explanation. Every reading loses on dev and
every reading wins on validation:

| ti_buy / ti_sell | dev net | dev Sharpe | dev PF | val net | val Sharpe | val PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 / 70 *(Table 1, literal)* | −$1,103 | −0.14 | 0.94 | +$916 | 0.52 | 1.15 |
| 60 / 40 | −$2,472 | −0.51 | 0.83 | +$1,744 | 0.99 | 1.35 |
| 65 / 35 | −$1,629 | −0.33 | 0.87 | +$1,331 | 0.84 | 1.37 |
| 70 / 30 *(buffer-zone reading)* | −$2,263 | −0.57 | 0.78 | +$2,008 | 1.33 | 1.87 |
| 80 / 20 | −$2,100 | −0.72 | 0.67 | +$1,009 | 1.18 | 1.90 |

Ten cells, five negative and five positive, split perfectly by *which split they
were run on* rather than by parameter. A parameter surface that is flat in the
parameter and steep in the sample period is measuring the sample period.

**On the LLM layer**, which cannot be run here, three things are visible from
the paper itself and are worth stating plainly:

1. **The filter is nearly inert by construction.** Buy valid if confidence
   > 0.25, sell valid if < 0.75. A model asked for a 0-1 confidence returns
   0.3-0.7 for almost everything, so most signals pass both gates. The paper
   confirms it: 1,717 trades become 1,401, and that 18% reduction is credited
   with +252 percentage points of return - implying the ~300 removed trades
   averaged catastrophic losses. That is a very strong claim for a filter this
   loose.
2. **The look-ahead argument does not hold.** The paper's defence is that the
   model is given no dates. But "Nonfarm Payrolls, actual 517K, forecast 185K"
   identifies January 2023 uniquely to anyone who has read the period - and a
   model trained through 2024 has. Withholding the date does not withhold the
   identity when the surprise itself is the fingerprint.
3. **It is an overlay on a backbone that has no edge.** Whatever the filter is
   worth, §3 says it is being applied to −$1.20 a trade.

## 6. Verdict on Paper A

**Not deployable.** Not "deployable with tighter risk", not "deployable on a
smaller size" - there is no edge to size.

- Pooled over 5.4 years and 212 trades: Sharpe 0.017, *t* = 0.04, p = 0.97.
- Negative at mid on the larger split, so better execution does not rescue it.
- The news calendar - the paper's entire premise - adds nothing on dev and
  everything on validation, which means it adds nothing.
- Trade frequency is 33-41 a year against the paper's 235. A denser calendar
  would raise n; it would have to also change the sign of the per-trade
  expectancy, and there is no reason in this data to expect that.

**The held-out test split (2025-07 to 2026-09) was not touched.** A strategy
that fails on dev and pools to zero on dev+validation has not earned the one
honest shot that split represents.

---

# PAPER B - SPAR intraday volume forecasting

## 7. The premise: these curves are not U-shaped

SPAR's argument is that pre-specifying the periodic shape - a U-curve fitted by
splines, as CMEM does - is a mistake. Here is the estimated curve for each
panel, no shape assumed, low activity `·` to high `@`:

```
USTEC  session |@*+++=+==-----::::::::....... .            .      ....: |  2.1x
USTEC  24h     |   .  :..=-:-:::.....:..::--:@%#**+++=====+==+=+==---   |  6.8x
XAUUSD session |%###*+@**++=+====-=---::::::..:....... . .... .:.    .  |  3.3x
XAUUSD 24h     |  ....-::-::-::::.:..:::=*+*=#@+++=--::-:.::::...  :..- |  8.2x
EURUSD session |#*****@#**++*++*##*==-----::::::.......      .      ..  |  5.7x
EURUSD 24h     |   ...=--#==+===--------++**+*@***=--::-:.:..:...  :..: | 59.1x
USDJPY session |%#***+@#**++++++*#*=-----::::::......    ...       .    |  5.7x
USDJPY 24h     |...:..=--*==+=---:-::---=+*#+#@*+*=--::-:.:..:.. :..*-- | 39.7x
```

Not one of the eight is U-shaped. Every US session here is **J-shaped** - heavy
at the open, decaying into a flat afternoon, because a CFD quote feed has no
closing auction to rebuild the right-hand arm of the U. The 24-hour grids are
**multi-humped**, with the Tokyo, London and New York peaks visible as separate
features and peak-to-trough ratios of 6.8× to 59×.

This is exactly the case the paper says a pre-specified curve cannot fit, on
data it never saw, and the premise survives contact with it: fitting any of
these eight with a U-spline would misspecify the periodic component badly
enough to poison everything downstream of it.

## 8. Forecast accuracy: the claim replicates, emphatically

Out-of-sample R² against the rolling historical mean, on logs where the models
are fitted. **Dev:**

| symbol | grid | SPAR1 | OLS1 | SPAR2 | OLS2 | SPAR3 | OLS3 | SPAR4 | OLS4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| USTEC | session | **0.834** | 0.713 | 0.808 | 0.693 | 0.833 | 0.712 | 0.808 | 0.683 |
| USTEC | 24h | 0.813 | 0.708 | 0.795 | 0.702 | 0.812 | 0.708 | 0.793 | 0.697 |
| XAUUSD | session | 0.756 | 0.614 | 0.730 | 0.616 | 0.756 | 0.613 | 0.730 | 0.615 |
| XAUUSD | 24h | 0.717 | 0.638 | 0.699 | 0.643 | 0.717 | 0.639 | 0.699 | 0.639 |
| EURUSD | session | 0.734 | 0.642 | 0.705 | 0.637 | 0.735 | 0.642 | 0.703 | 0.626 |
| EURUSD | 24h | 0.562 | 0.428 | 0.540 | 0.479 | 0.566 | 0.431 | 0.545 | 0.486 |
| USDJPY | session | 0.712 | 0.617 | 0.682 | 0.619 | 0.712 | 0.616 | 0.682 | 0.617 |
| USDJPY | 24h | 0.612 | 0.513 | 0.593 | 0.543 | 0.615 | 0.515 | 0.597 | 0.549 |

**Validation** repeats it - SPAR1 0.894 vs OLS1 0.845 on USTEC session, 0.842
vs 0.755 on the 24-hour grid, and so on down every row.

**SPAR beats OLS on identical features in 64 of 64 comparisons** - 4 specs × 8
panels × 2 splits, no exceptions. The margin is 5 to 14 R² points. This is as
clean a replication as this project has produced for any paper.

Two footnotes that a deploying desk needs:

**The level R² is unstable, and it is the volatility feature that breaks it.**
The paper writes its R² on levels while fitting on logs, and
`exp(E[log V]) ≠ E[V]`. Every panel is fine on specs 1 and 2. On USDJPY the
specs carrying the OK volatility block collapse: level R² of **−27.8** (SPAR3)
and **−34.3** (OLS3) on dev, **−59.5** and −4.8 on validation, against 0.59-0.64
for the same panel on spec 1. The estimator occasionally throws a large
volatility reading, the exponential turns it into an enormous volume forecast,
and one such day destroys a variance ratio. This is not a broken model - it is a
missing smearing correction, and it hits SPAR and OLS alike. Anyone reading a
*level* forecast out of this owes it one; the log forecasts are unaffected.

**The extra features barely earn their place.** SPAR1 (six intraday lags) ties
or beats SPAR3 and SPAR4 nearly everywhere. The volatility block and the 22-day
cross-day block add ~0.00 to 0.01. The within-transformation is doing all the
work, which is at least the paper's own point.

## 9. VWAP replication: where the claim stops

The paper's economic argument runs through the dynamic VWAP schedule of
Bialkowski et al. (2008). Mean absolute tracking error, and - the part the paper
omits - a **paired stationary block bootstrap** of the difference on common
days. Negative `diff` means the left schedule tracked closer.

**SPAR3 vs OLS3** - the within-transformation itself:

| symbol | grid | dev diff (bp) | dev p | val diff (bp) | val p |
| --- | --- | ---: | ---: | ---: | ---: |
| USTEC | session | **−0.132** | **0.000** | +0.157 | 0.156 |
| USTEC | 24h | +0.098 | 0.033 | +0.467 | 0.059 |
| XAUUSD | session | +0.058 | 0.141 | +0.149 | 0.050 |
| XAUUSD | 24h | −0.048 | 0.293 | −0.018 | 0.673 |
| EURUSD | session | +0.104 | 0.007 | +0.045 | 0.125 |
| EURUSD | 24h | −0.050 | 0.084 | −0.015 | 0.679 |
| USDJPY | session | +0.154 | 0.000 | +0.156 | 0.000 |
| USDJPY | 24h | +0.063 | 0.116 | +0.027 | 0.618 |

Across both splits: SPAR closer in **1** of 16 panels at p < 0.05, OLS closer in
**4**, indistinguishable in **11**. The 64-of-64 forecasting advantage
transmits to the execution objective essentially not at all. The same comparison
on spec 1 gives 2 / 3 / 11, and on spec 4 it is worse still - **0 / 7 / 9**,
OLS4 significantly closer in seven panels.

**SPAR3 vs equal weight** - is forecasting worth anything at all? A qualified
yes: significantly better in 6 of 8 dev panels (−0.6 to −1.4 bp) and 2 of 8
validation panels, and significantly *worse* on USDJPY 24h in both. So a
forecast-driven schedule does beat a naive one, by around 1 bp, inconsistently,
and on one instrument it is actively harmful.

**Why the accuracy does not transmit.** The dynamic schedule already contains a
static component - the trailing 22-day slot mean - which *is* the periodic
curve. SPAR's advantage over OLS is precisely its handling of periodicity. The
schedule has therefore already absorbed most of what SPAR is better at before
the intraday forecast is consulted, and the marginal gain is small by
construction. That is a structural reason, not a sampling accident, and it
predicts the result seen here.

**On the paper's own numbers.** It reports SPAR3 7.013 bp against OLS3 7.275 bp
and monetises the 0.262 bp gap at ~$14,400 a year. It gives no interval. The
day-to-day standard deviation of tracking error in this sample is 4-11 bp -
twenty to forty times that gap. That does not prove their difference is noise;
their 2,376 days may well resolve it. It does mean the difference is small
enough that it *needs* an interval, and none is given.

## 10. Verdict on Paper B

**The forecasting model is deployable; the advertised execution savings are
not.**

Deploy the model where it is strong - as an intraday liquidity nowcast, a
participation-rate governor, or a slot-level activity forecast. It is a
20-line estimator, it needs no tuning, it handles multi-hump 24-hour
periodicity without being told the shape, and it beat its own control 64 times
out of 64.

Do not underwrite a guaranteed-VWAP book on the strength of the tracking-error
table. The gain over an untransformed model is not reliably present, and the
gain over equal weight is ~1 bp and inconsistent across instruments.

**The test split was not spent here either.** Both halves of the claim resolved
consistently on dev and validation - the forecasting half confirmed twice, the
execution half rejected twice - so there is no open question the final split
would settle.

---

## 11. Reproducing this

```bash
python scripts/backtests/backtest_news_breakout.py                    # Paper A, dev + validation
python scripts/backtests/backtest_news_breakout.py --audit-only       # just the calendar audit
python scripts/backtests/backtest_news_breakout.py --sweep            # threshold surface
python scripts/backtests/backtest_volume_spar.py                      # Paper B, all four instruments
python scripts/backtests/backtest_volume_spar.py --symbols USTEC --splits dev
python -m pytest tests/test_macro_calendar.py tests/test_volume_spar.py
```

Code: `src/qlab/macro_calendar.py`, `src/qlab/strategies/news_breakout.py`,
`src/qlab/volume_spar.py`. Trade tapes and result frames land in
`reports/strategies/newsbreakout_*.parquet` and `spar_*.parquet`.

The look-ahead guard worth knowing about is
`test_forecasts_do_not_move_when_the_future_is_replaced`: it rewrites every
observation after day 200 with noise and asserts that no forecast before day 200
changes. A within-transformation is an easy place to leak - use the mean
*including* day *t* and the target sits inside its own predictor - and that test
is what stands between §8 and a very impressive artefact.
