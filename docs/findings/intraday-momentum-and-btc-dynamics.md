# Two SSRN papers, evaluated

Subject: two papers handed over together, with instructions to extract the
strategy each proposes and backtest it.

1. **Zarattini, Aziz and Barbon (2024/2025)**, *Beat the Market: An Effective
   Intraday Momentum Strategy for S&P500 ETF (SPY)* (SSRN 4824172). A complete,
   closed rule set. On SPY 1-minute bars from May 2007 to April 2024 it reports
   1,985% total return, 19.6% annualised, 14.3% volatility, **Sharpe 1.33**,
   25% maximum drawdown, 43% daily hit ratio, alpha 19.6%/yr, beta −0.07.
2. **Eross, McGroarty, Urquhart and Wolfe (2017)**, *The Intraday Dynamics of
   Bitcoin* (SSRN 3013699). **This paper proposes no trading strategy.** It is a
   descriptive microstructure study of BTC-e 5-minute data, 1 Nov 2014 to
   31 Oct 2016, closing on the hope that its findings "will be of great interest
   to ... Bitcoin investors who could use this information in their trading
   strategies". No rule, no entry, no exit, no backtest.

Method: paper 1 is implemented in `src/qlab/strategies/intraday_momentum.py` and
run on **USTEC** (Nasdaq-100 CFD, the corpus's nearest thing to SPY) with gold
and two FX majors as breadth checks, over dev (2020–2023, 959 sessions),
validation (2024–H1 2025, 366) and one final pass on the locked test split
(H2 2025–Aug 2026, 286). Paper 2 is implemented in
`src/qlab/strategies/btc_intraday.py` and run on **Binance spot BTCUSDT 5-minute
klines, Jan 2020 – Aug 2026, 569,490 bars** — a different exchange, a different
decade, and a 3.3× longer sample than the original.

---

## Verdict

> **Paper 1 replicates in mechanism and decays in magnitude.** On USTEC the
> rules produce Sharpe **1.13** on dev against a buy-and-hold 0.69, with alpha
> +17.0%/yr (t = +2.51) and beta −0.04 — against the paper's own 19.6% and
> −0.07. It then falls to **0.84** on validation and **0.44** on the held-out
> test split, where it is no longer distinguishable from zero (t = +0.47). All
> 24 dev configurations are positive on USTEC and the edge is absent on gold and
> both FX majors, which is the instrument-specificity the paper's own mechanism
> predicts. This is a real effect that is weaker than advertised, not a fake one.
>
> **Paper 2's activity findings replicate, its return finding does not, and its
> spread finding is an artefact of its own estimator.** The volume and
> volatility clocks are still there a decade later on a different exchange. The
> claim that returns are highest 08:00–16:00 GMT does not survive a test
> (difference vs the rest of the day: t = +0.08, p = 0.94). Both strategies the
> paper's findings imply are untradable — one because the pattern is not there,
> the other because the pattern **is** there and is 30–40× smaller than a
> Binance spot fee.

---

## 0. What this corpus can and cannot say about these papers

**Paper 1's instrument is not SPY.** This corpus holds no US equity ETF. USTEC
is a Nasdaq-100 CFD: same 09:30–16:00 New York session, same dealer-gamma and
index-arbitrage plumbing, more volatile and more concentrated underlying. A
strategy that survives here has cleared a *different* bar, not a lower one.

**Paper 1's costs are not the paper's costs.** $0.0035/share commission and
$0.001/share slippage are US equity terms and meaningless on a CFD. Spread is
measured per hour from the tick corpus, commission is the published contract
term, slippage is measured latency drift — all through the project's standard
`CostModel`, so this strategy is costed on the same basis as everything else
here. The bill comes to **1.5 bps/day, 22% of gross**.

**The CFD has no opening auction.** SPY's 09:30 print is a crossing auction
after a 17.5-hour halt; USTEC has been trading all night. The gap adjustment
therefore measures overnight drift rather than auction imbalance, and early-
session boundaries are narrower here. That makes early entries *easier* to
trigger, so the first decision time was swept rather than assumed.

**Paper 2's exchange no longer exists.** BTC-e was seized in July 2017. Binance
spot is the substitute, and that is the point of the exercise rather than a
compromise in it: F1–F4 are claims about *Bitcoin's clock*, and a clock that
only existed on one defunct venue in 2015 is not what the paper argues for.

**Paper 2's volume is real volume.** Unlike everywhere else in this project,
where "volume" means quote updates, Binance klines carry genuine traded size —
so the volume findings are tested on the quantity they were stated about.

---

## 1. Paper 1 — the Noise Area strategy

### 1.1 The rules, as transcribed

For each time-of-day τ in the session, σ is the mean absolute move from the open
over the previous 14 sessions **at that same time-of-day**:

```
move[t-i, τ]  = | close[t-i, τ] / open[t-i, 09:30] − 1 |,   i = 1..14
σ[t, τ]       = mean over i
upper[t, τ]   = max(open[t, 09:30], close[t-1, 16:00]) × (1 + σ[t, τ])
lower[t, τ]   = min(open[t, 09:30], close[t-1, 16:00]) × (1 − σ[t, τ])
```

Decisions fire **only at HH:00 and HH:30** — entries, reversals and stops alike.
Close beyond a boundary opens a position in that direction. The trailing stop is
`max(upper, VWAP)` for a long and `min(lower, VWAP)` for a short. Everything is
flat at 16:00. Size is `AUM × min(4, 2% / σ_daily) / open`, where `σ_daily` is
the sample standard deviation of the previous 14 daily returns.

Because no rule ever looks *inside* a bar, this runs on bars with no intrabar
ordering assumption — unusually clean for an intraday strategy.

The band is strictly point-in-time (day `t` uses only `t−1 … t−14`), and that is
pinned by test, not by inspection: a single 10% session must not widen its own
band. Two implementation bugs were caught this way and are now regression tests —
a polars `Int8` overflow silently turning every time-of-day past 02:00 negative,
and a session-boundary pairing issue in the spread estimator.

### 1.2 Results on USTEC

| split | sessions | CAGR | vol | Sharpe | MDD | hit | t | legs/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev (2020–2023) | 959 | +16.40% | 14.3% | **1.13** | −16.4% | 26.0% | +2.20 | 0.84 |
| validation (2024–H1 25) | 366 | +11.26% | 13.8% | **0.84** | −6.4% | 23.2% | +1.01 | 0.83 |
| **test (H2 25–Aug 26)** | 286 | +5.87% | 15.5% | **0.44** | −11.1% | 23.1% | +0.47 | 0.86 |
| *paper, SPY 2007–2024* | *4,280* | *+19.6%* | *14.3%* | *1.33* | *−25%* | *43%* | — | *1.79* |

Against holding the index over the same sessions: dev Sharpe 0.69, validation
1.01, test 1.25. The strategy beats buy-and-hold on dev, loses to it on
validation and test — but with a beta of −0.04 it is not trying to track it.

| split | alpha /yr | t | beta | t | bootstrap Sharpe 95% CI | sign-placebo |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| dev | **+16.96%** | **+2.51** | −0.041 | −1.43 | [+0.30, +1.92] | beats 97.8% |
| validation | +11.24% | +1.07 | +0.016 | +0.15 | [−0.71, +2.06] | beats 83.8% |
| test | +10.90% | +0.63 | −0.170 | −1.47 | [−1.87, +2.01] | beats 70.0% |

The **sign placebo** holds the trade calendar fixed — same sessions, same entry
and exit times, same leverage, same costs — and randomises only the *direction*
of each leg. On dev the real Sharpe beats 97.8% of 400 such draws, so the edge is
in the directional call rather than in happening to be in the market at good
times. That evidence weakens monotonically alongside everything else.

### 1.3 What replicates precisely

**The trade-level signature.** The paper's case for the VWAP trail is that it
converts a symmetric loser into a skewed winner. That is exactly what the tape
shows, in all three splits:

| split | stops | mean | hit | closes | mean | hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 527 | −22.8 bp | 16.3% | 279 | **+58.2 bp** | 88.5% |
| validation | 206 | −20.3 bp | 17.0% | 96 | **+59.2 bp** | 87.5% |
| test | 174 | −15.5 bp | 20.7% | 71 | **+45.6 bp** | 91.5% |

Two thirds of legs are small losses; the third that reaches the bell is worth
2–3× as much. Test-split best/worst day is **+11.10% / −2.35%**. The paper's
positive skew is the most robust thing in the whole study.

**The volatility target does its job.** Realised volatility is 14.3% on dev
against the paper's 14.3% — the scaler pins the risk level, and the strategy's
vol stays in a 13.8–15.5% band across six years and three regimes at a mean
leverage of 1.76×.

**The stop ordering.** The paper builds the strategy in three steps and reports
all three. Same ordering here: the base model (trailing at the *opposite* band)
has the highest hit rate (33–36%) and by far the worst drawdown (−34%); tightening
to the current band raises Sharpe from 0.77 to 1.16 and cuts drawdown to −16%.

**Instrument specificity.** Median Sharpe across all 24 dev configurations:

| instrument | median Sharpe | best | median t |
| --- | ---: | ---: | ---: |
| **USTEC** | **+0.78** | +1.16 | +1.52 |
| XAUUSD | +0.08 | +0.63 | +0.16 |
| USDJPY | −0.16 | +0.38 | −0.32 |
| EURUSD | −0.34 | +0.13 | −0.68 |

All 24 USTEC configurations are positive; the result does not depend on picking
the right corner of the grid. And the effect is absent on gold and FX, which is
what a story about equity-index demand/supply imbalance predicts. A version of
this that worked equally well on EURUSD would be *more* worrying, not less.

### 1.4 What does not replicate

**The hit ratio.** 23–26% here against the paper's 43%, and 0.84 legs/day against
its 1.79. The bands on USTEC bind less often and hold longer once entered. The
expectancy arrives the same way — it is simply concentrated into fewer, larger
trades.

**The magnitude, and it decays monotonically.** Sharpe 1.13 → 0.84 → 0.44, alpha
t-statistic 2.51 → 1.07 → 0.63. Calendar-year net returns show where it comes
from:

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to Aug) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| +20.6% | −3.3% | +17.9% | +29.6% | +9.2% | +13.3% | +0.6% |

Six of seven years positive, but the two best are 2020 and 2022 — the two
high-volatility years. This is a trend-following strategy and it earns when
there are trends. Nothing here is evidence of a break; it is evidence of a
strategy whose payoff depends on a regime, evaluated across a stretch that
contained fewer of them.

### 1.5 Reading

The paper's rules, transcribed without modification and costed properly, produce
on the nearest available instrument an effect with the right sign, the right
skew, the right risk level, the right instrument-specificity and roughly the
right alpha — and a t-statistic that falls below 1 on held-out data. Sharpe 1.33
is not reproduced; nothing in six years of USTEC contradicts a true Sharpe
somewhere between 0.4 and 1.1, which is a materially different proposition.

---

## 2. Paper 2 — Bitcoin intraday dynamics

Since the paper states no strategy, the work splits in two: replicate what it
actually claims, then test the strategies those claims imply. Both strategies
were specified before being run.

### 2.1 The descriptive claims, on a different exchange a decade later

| # | claim | verdict |
| --- | --- | --- |
| F1 | Volume low until 07:00 GMT, rises, peaks ~14:00, decays — an n-shape | **confirmed** |
| F2 | RV highest 07:00–18:00 GMT, declines after | **partial** |
| F3 | Spread n-shaped and closely tracking RV | **confirmed but contaminated** |
| F4 | Returns highest 08:00–16:00 GMT | **rejected as stated** |
| F5 | Returns ⟂ volume/RV negative, ⟂ spread positive; bilateral Granger | **confirmed in sign** |

**F1 is the strongest result in the paper and it holds.** Volume troughs at
04:00–05:00 GMT (206 BTC per 5-min bar) and peaks at 14:00 (417), a 2.0× swing,
with the 07:00–18:00 block running 1.32× the rest of the day. Trade count traces
the same shape (6,315 → 13,033). Ten years, a different venue, an asset that grew
a hundredfold — and Bitcoin still keeps European and North American office hours.

**F2 is weaker than claimed.** RV over 07:00–18:00 is 1.18× the rest, peaking at
13:00 GMT. But it does *not* cleanly decline after 18:00 — the 20:00–22:00 block
runs at 4.97e−6, above the daily mean. The daytime hump is real; the "and then it
subsides" half is not.

**F4 fails the test it was never given.** Across 1,972 days:

| window | mean | t | p | annualised |
| --- | ---: | ---: | ---: | ---: |
| inside 08:00–16:00 (8 h) | +6.45 bp/day | +1.54 | 0.124 | +23.5% |
| outside (16 h) | +5.98 bp/day | +1.11 | 0.265 | +21.9% |
| **difference** | **+0.46 bp/day** | **+0.08** | **0.94** | — |

The eight-hour window and the sixteen-hour remainder earn the same. A weaker
version does survive — per *hour* inside the window is about 2.2× as productive —
but that is a statement about efficiency of exposure, not about when to be long,
and the paper's own figures never test it.

**F5 replicates in every sign.** corr(ret, volume) = −0.0165, corr(ret, RV) =
−0.0035, corr(ret, spread) = +0.0196, corr(volume, RV) = +0.182. Every sign
matches. Every magnitude is around 0.02 — three hundredths of one percent of
variance. Granger causality is bilateral and significant for every pair at 7
lags, exactly as reported; at n = 569,490 that is nearly uninformative, and the
weakest relationship (volume → ret, F = 6.20) carries p < 3e−7 while explaining
essentially nothing.

### 2.2 F3 is largely an artefact of the estimator

The paper has no quotes, so it estimates the spread with Corwin–Schultz (2012)
from consecutive high-low ranges, then reports that the spread is n-shaped and
tracks RV — attributing this to the absence of market makers.

This corpus **does** carry quoted spreads, so the estimator can be checked
against the truth on the same 5-minute bars it is used on:

| instrument | CS est | true quoted | ratio | corr per bar | corr by time-of-day | corr(est, range) | corr(true, range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| USTEC | 2.69 bp | 0.63 bp | **4.3×** | +0.19 | +0.23 | **+0.46** | +0.35 |
| XAUUSD | 1.78 bp | 0.61 bp | **2.9×** | +0.06 | +0.26 | **+0.35** | +0.17 |
| EURUSD | 0.87 bp | 0.16 bp | **5.4×** | −0.07 | **−0.48** | **+0.37** | −0.12 |

At 5-minute frequency the estimator overstates the spread by three to five times
and, in every instrument, correlates **better with the high-low range than with
the actual spread** — on EURUSD it correlates *negatively* with the truth at the
exact aggregation the paper plots. Given a spread held constant by construction,
its per-bar estimates still scatter by more than half their own level.

This matters directly. RV is a squared return and Corwin–Schultz is built from
the range; on the BTC sample they correlate +0.850 across hours, and "RV Granger-
causes the spread" is the single strongest relationship in the paper's Table 3
(F = 7,487 here). Two functions of the same range moving together is not evidence
about market makers. F3's shape is probably real — spreads do widen when
volatility rises — but the paper's own instrument cannot establish it, and the
mechanism it infers is not identified.

Corwin–Schultz was designed for **daily** equity data. Using it at 5 minutes is
outside its design range, and this is what that costs.

### 2.3 Strategy A — long 08:00–16:00 GMT (from F4)

| split | gross Sharpe | buy & hold | net Sharpe | gross bp/day | cost bp/day |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 0.54 | 0.56 | **−1.90** | +5.91 | 26.5 |
| validation | 0.98 | 1.24 | **−2.07** | +7.89 | 24.5 |
| test | −0.39 | −0.62 | **−3.48** | −2.93 | 23.5 |

Gross, the window earns roughly the same risk-adjusted return as holding around
the clock, at 57% of the volatility and a −51.6% drawdown against −84.8%. That is
the most that can be said for it, and it is not statistically distinguishable
from either zero or buy-and-hold in any split.

Net, it is a catastrophe. Being flat overnight costs two round turns a day: at
Binance spot standard tier (10 bp/side) plus the estimated half-spread, that is
**~25 bp/day against a gross edge of ~6 bp/day**. Breakeven needs a round turn
under 3 bp; the fee alone is 20. A sweep over 24 window choices finds no window
that is net positive and none that singles out 08:00–16:00 — the best gross
Sharpe belongs to 06:00–24:00, which is nearly buy-and-hold.

### 2.4 Strategy B — fade the last bar when it was unusual (from F5)

Position at bar *t* is `−sign(ret[t−1])`, taken only when bar *t−1*'s volume (or
RV, or trade count) exceeded its own trailing daily quantile. The unconditional
version is the control.

| split | conditioned (volume, q=0.9) | t | unconditional | t |
| --- | ---: | ---: | ---: | ---: |
| dev | **+0.725 bp/bar** | +4.26 | +0.357 bp/bar | +10.66 |
| validation | **+0.506 bp/bar** | +2.63 | −0.049 bp/bar | −1.29 |
| test | +0.073 bp/bar | +0.40 | +0.020 bp/bar | +0.53 |

**F5's specific claim is vindicated on validation.** Plain 5-minute reversal died
out of sample — the unconditional edge went negative — while conditioning on high
volume kept a positive, significant edge. The volume term carries genuine
information; that is the paper's contribution and it survives a real out-of-sample
test. Gross Sharpe of the conditioned rule reaches 2.25–3.32.

And it is completely untradable. **Breakeven is a round turn of 0.25–0.36 bp.
Binance spot taker alone is 10 bp** — thirty to forty times too expensive, before
the spread. Net Sharpe across every signal, every quantile and every split runs
from −26 to −242, and net CAGR is −100% everywhere. On the test split the effect
is gone anyway (t = +0.40).

This is the standard fate of a high-frequency statistical regularity: real,
measurable, significant, and living entirely inside the bid-ask spread. The paper
was right not to claim a strategy.

---

## 3. What was actually built

| file | what |
| --- | --- |
| `src/qlab/strategies/intraday_momentum.py` | Paper 1's rules, three stop variants, vol targeting, sign placebo |
| `src/qlab/strategies/btc_intraday.py` | Paper 2's variables, profiles, cross-correlation, Granger, both strategies |
| `src/qlab/microstructure.py` | Corwin–Schultz, realised variance, and the estimator-vs-truth check |
| `scripts/pipeline/fetch_btc_binance.py` | Binance monthly dumps → `data/external/binance` (80 months) |
| `scripts/backtests/backtest_intraday_momentum.py` | Sweep / detail / locked test runner |
| `scripts/backtests/backtest_btc_intraday.py` | `replicate` / `estimator` / `strategy` / `confirm` / `test` |
| `tests/test_intraday_momentum.py` | 24 tests: point-in-time bands, gap adjustment, decision timing, stop precedence, sizing |
| `tests/test_microstructure.py` | 13 tests: estimator algebra, degenerate inputs, session-boundary behaviour |

BTC data is kept in `data/external/binance`, deliberately outside
`data/processed` — that tree is the broker tick corpus under one cleaning policy,
and exchange klines are not it.

Full suite: **513 tests, all passing.**

## 4. If this were taken further

**Paper 1 is worth more work; paper 2 is not.** Three things would sharpen the
paper 1 verdict:

1. **Get SPY or ES.** The single largest source of uncertainty is that USTEC is
   not the instrument. A 2007-start SPY minute series would test the paper on its
   own terms and add the 2008 and 2011 volatility regimes that this corpus lacks
   entirely.
2. **Condition on the regime rather than lament it.** 2021 was the one losing
   year and the two best were the two most volatile. The paper itself tests
   whether dealer gamma imbalance predicts profitability; a VIX- or realised-vol-
   conditioned version is the obvious next cut, and this corpus can do it.
3. **Test the semi-hourly grid.** Restricting decisions to HH:00 and HH:30 is
   presented as noise control but never justified. A 15-minute or 10-minute grid
   would separate the claim from the convenience.

For paper 2, the honest next step is not another strategy — it is a maker-fee or
futures-basis venue where a 0.7 bp/bar edge could conceivably clear costs, and
even there the effect is gone by the test split.
