# EURUSD: twelve rejected hypotheses, and no strategy

**Verdict: rejected. No EURUSD strategy is proposed, and none should be
deployed on the strength of this corpus.**

Twelve hypothesis families were tested. All twelve failed. This is not a report
about running out of ideas - it is a report about two structural facts that make
the search close to hopeless on this instrument, and one measurement that came
out of it and is worth keeping.

The two facts:

1. **There is no risk premium to fall back on.** Over dev, EURUSD returned
   **+0.25% in four years** - a CAGR of 0.06% at a Sharpe of 0.01. The USTEC
   study could end in an accepted strategy because there was a positive-drift
   asset to hold and the work was in sizing it. Here there is nothing to size. A
   EURUSD strategy has to be pure alpha or it is nothing.
2. **Per unit of the volatility it has to beat, EURUSD is one of the dearest
   instruments in the corpus, not the cheapest.** Stage 2 ranked it cheapest at
   0.586 bps a round turn. Divided by the move a strategy actually has to
   capture, that ranking inverts - see §2.

- Research driver: `scripts/research/eurusd_research.py` (re-runs every rejection below)
- Measurement kept: `src/qlab/rollover.py`, tested in `tests/test_rollover.py`
- Study page: https://claude.ai/code/artifact/79a2c46e-c204-4c86-a3c9-d77b5bdbc1d2

---

## 1. The rejection log

Each family was written down before its number was read, and each is reported
whatever it said. The hurdle is a **0.586 bps** round turn on dev (0.582 on
validation) - and note its composition, because it decides everything below:
spread is **5%** of it, commission **78%**, slippage 17%. On a raw-spread
account EURUSD's quoted spread is zero more than half the time. The cost is
almost entirely a fixed charge that does not shrink when the market goes quiet.

| # | hypothesis | result | verdict |
| --- | --- | --- | --- |
| 1 | Hour-of-day drift shape | dev vs validation correlation **+0.21** (60m), +0.14 (30m) | weak, and see #12 |
| 2 | Intraday window selection (dev-select, val-verify) | dev SR +0.13 to +0.48, **validation SR -3.12 to -4.21** | decisive failure |
| 3 | Minute-level autoregression | reversal real at t = -20, but the implied edge peaks at **0.19x** of a round turn | too small |
| 4 | USD-factor residual reversion (stat arb) | contemporaneous R² 28%; best gross edge **0.54 bps** against 0.586 cost | just under |
| 5 | Residual reversion, extreme tail | cells that clear cost in validation are **uniformly negative in dev** | noise |
| 6 | Daily momentum, 5 / 20 / 60 / 120 / 250 day | signs flip dev→validation on 3 of 5; nothing past t = 1.1 | no |
| 7 | Daily reversal (fade yesterday) | dev -2.70 bps, validation +0.73 bps | sign flips |
| 8 | Day of week | flips on 3 of 5 weekdays; best t = 1.14 | no |
| 9 | Volatility regime → next-day return | dev +0.15 / -0.58 / -0.53, validation -1.54 / +3.51 / +3.43 | sign flips |
| 10 | Month-end fix effect | n = 47 in dev, 17 in validation | underpowered |
| 11 | Session legs (Asia / London / overlap / late) | signs flip on 3 of 4 legs | no |
| 12 | Rollover-window drift | +1.19 bps/night tradable, t = 5.05, **stable in both splits** - and it is the carry | see §3 |
| — | Session opening-range breakout | previously evaluated: **-490 R over 9,622 trades**, the worst of the four symbols | already rejected |

### The two negatives worth reading

**#2 is the cleanest.** EURUSD's intraday shape correlates +0.21 between dev and
validation, against USTEC's -0.11, so the family deserved a real test rather
than an assumption. Rank the 24 UTC hours by dev t-statistic, take everything
past a threshold, and run that fixed selection forward:

```
|t| > 1.0   IN dev  SR +0.12     OUT val  SR -3.37
|t| > 1.5   IN dev  SR +0.31     OUT val  SR -3.12
|t| > 2.0   IN dev  SR +0.48     OUT val  SR -4.21
```

Worse out-of-sample than the same test on USTEC. And the hours that survive the
dev filter at |t| > 2 are 1, 19, **21 and 22** - the roll. The only part of the
intraday shape that is stable across both periods is the thing §3 disqualifies.

**#3 and #4 fail the same way, and it is worth being precise about how.**
Reversal in EURUSD minute bars is not in doubt: t reaches -20 over 1.3 million
observations. But a t-statistic measures whether an effect is distinguishable
from zero, not whether it is bigger than a cost. The best cell implies an
expected edge of 0.109 bps; a round turn is 0.586. The stat-arb residual does
better - the other three symbols explain 28% of EURUSD's minute variance, and
the unexplained part does revert - but its best gross edge is 0.54 bps, still
short of the 0.586 it has to clear before anything is left.

---

## 2. The cost ranking inverts

Stage 2's headline was that a round turn costs 0.515 to 0.689 bps across the
four symbols - a 1.3x band - and concluded that cost is not the reason to pick
one instrument over another. Measured against the volatility a strategy has to
capture, that conclusion needs an amendment:

| symbol | round turn (bps) | 1m sd (bps) | daily sd (bps) | rt / 1m sd | rt / daily sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 0.585 | 1.406 | 49.4 | 0.42 | 0.0119 |
| USDJPY | 0.724 | 1.501 | 59.2 | **0.48** | **0.0122** |
| XAUUSD | 1.092 | 2.704 | 96.9 | 0.40 | 0.0113 |
| USTEC | 1.209 | 4.493 | 167.5 | **0.27** | **0.0072** |

**In bps EURUSD is the cheapest and USTEC the dearest. Per unit of opportunity
the order reverses**, and by a wider margin than the bps figures ever suggested:
USTEC is 1.6x cheaper than EURUSD at minute scale and 1.65x cheaper at daily
scale. The reason is that commission is a fixed 0.455 bps and EURUSD moves a
third as much as an index does, so the same charge eats a much larger share of
every move.

This is the same shape of correction Stage 2 applied to Stage 0: a cost figure
that looked comparable across instruments stops being comparable once it is
divided by the thing it has to be paid out of. It also explains the two studies'
outcomes better than either study's own detail does.

---

## 3. The rollover drift, and why it is not an edge

This was the one thing in the whole study with a stable sign, and it survived
costs. Buy at the ask at 20:55 UTC, sell at the bid at 23:55 UTC, straddling the
broker's daily roll with both fills in an hour whose spread is normal:

| | n | tradable | net of commission | t | hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 807 | +1.187 bps | **+0.733 bps/night** | +5.05 | 55.5% |
| validation | 295 | +0.958 bps | **+0.498 bps/night** | +1.76 | 52.2% |

That is roughly a 1.7 Sharpe on dev and 0.9 on validation, on a position held
three hours a day. It looked like the answer.

**It is the carry.** Spot FX settles T+2, so at each roll the value date moves
forward and the quote adjusts by the tom-next forward points - which *are* the
interest differential. The broker then charges swap to offset it. Capturing the
price drift requires holding through the exact moment the offsetting charge
lands.

The test that settles it is the sign on instruments whose carry runs in opposite
directions. Over this corpus the ECB sat below the Fed throughout, so a long
EURUSD **pays** carry; JPY sat at or below zero throughout, so a long USDJPY
**receives** it; gold yields nothing and costs financing, so a long **pays**.
Carry compensation therefore predicts: EURUSD up, USDJPY **down**, XAUUSD up.

| symbol | carry on a long | predicted drift | dev | validation | agrees |
| --- | --- | ---: | ---: | ---: | :---: |
| EURUSD | pays | up | +1.347 (t 5.05) | +0.973 (t 1.76) | yes |
| USDJPY | receives | **down** | **-1.467** (t -5.89) | **-1.438** (t -2.51) | yes |
| XAUUSD | pays | up | +2.564 (t 3.85) | +2.951 (t 2.38) | yes |

Six of six. The one instrument whose long *receives* carry is the one that
drifts down, in both periods, at t = -5.89 and -2.51. There is no version of
"transient flow imbalance" that produces that pattern; there is exactly one
version of "this is the roll" that does.

So the trade earns +0.73 bps of price and pays an unmeasured swap that theory
says is the same number with a broker markup on top. A strategy whose entire
P&L is the difference between a measured quantity and an unmeasured one that is
predicted to equal it is not a strategy.

**What is worth keeping is the measurement.** `qlab.rollover` computes the price
half of the overnight basis for any symbol at tradable prices, and
`basis_table()` runs the sign test above. Stage 2 listed overnight swap under
"not modelled"; this does not close that gap, but it narrows it - the price leg
is now measured, and only the charge is missing.

One thing it changes elsewhere: the USTEC risk-managed long holds overnight and
its report treats swap as a pure cost with a break-even of about 5 bps a night.
USTEC halts across the roll, so it has no drift of this kind to offset the
charge - but any *future* strategy on a quoting instrument does, and costing its
swap without crediting the basis would overstate the drag by 1 to 3 bps a night.

---

## 4. Where the test split was read

No strategy was selected here, so there was nothing to run a final evaluation
on, and the locked split was not used for one. It was read once, for the
rollover drift by calendar year, before that effect was rejected:

| year | n | mid drift | t |
| --- | ---: | ---: | ---: |
| 2025 (test portion) | 103 | +2.524 bps/night | +4.88 |
| 2026 | 137 | +0.424 bps/night | +0.56 |

Recorded because it happened, not because it matters: the statistic was rejected
on the cross-instrument sign test, nothing was fitted to it, and nothing is being
deployed. A future EURUSD study should still treat the rollover-drift statistic
specifically as having been seen on test.

---

## 5. What would change the answer

In rough order of how much would have to be true:

- **A live swap reading.** The rollover basis is +0.73 bps a night at tradable
  prices. If a particular account's EURUSD swap on a long is genuinely smaller
  than that, the trade is positive - and `SYMBOL_SWAP_LONG` is readable from
  MT5 in one line. This is the only idea in the study that could be settled by a
  measurement rather than by more data. The prior should be that brokers do not
  leave that gap open, but it costs nothing to look.
- **More data, for the daily families.** Dev holds about a thousand sessions at
  a 49 bps standard deviation, so a strategy needs a Sharpe near 1.0 to reach
  t = 2. Several daily hypotheses sit at t between 0.5 and 1.1 with the right
  sign in one period - which is exactly what both a real weak edge and pure
  noise look like at this sample size. Nothing here distinguishes them, and
  another pass over the same data will produce a false positive rather than an
  answer.
- **Options, for the volatility premium.** EURUSD volatility clusters like
  everything else, but with no drift to size there is nothing for that forecast
  to do. The FX volatility risk premium is real and this account cannot reach
  it.

What would *not* change the answer is searching harder. The honest reading of
hypothesis #2 is that this corpus will hand over a positive in-sample result to
anyone who asks it enough questions - it gave one an SR of +0.48 - and that the
same result is worth an SR of -4.21 the moment it is asked to work on data it
has not seen.

**Recommended instead:** the same overlay machinery applied to XAUUSD, which has
what EURUSD lacks - a real drift (dev Sharpe 0.45 on a +30.8% four-year move,
validation Sharpe 2.19 on +60.4%, though that second window is one exceptional
gold market and should not be read as a expectation) and the second-best
cost-per-volatility figure in the table above. `risk_managed_long`
is already symbol-parameterised and its tests run against any symbol.
