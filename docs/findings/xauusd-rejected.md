# XAUUSD: the instrument whose return was mostly financing

**Verdict: rejected. No XAUUSD strategy is proposed.** Ten hypothesis families
were tried, independent of the session opening-range breakout evaluated
earlier. All ten failed, and one measurement explains most of why.

**Gold's dev return is not gold's return.** Of a headline +27.4% over 4.8 years,
**+2.0% accrued in liquid hours** — 0.42% a year. The rest is the roll and the
halt gap, which is the financing adjustment a long earns in the price and hands
straight back in swap. At the measured basis of 2.56 bps a night, holding gold
through dev **loses 4.47% a year**.

- Research driver: `scripts/research/xauusd_research.py` (re-runs every rejection)
- Measurement added: `qlab.rollover.drift_decomposition`, tested in `tests/test_rollover.py`
- **The locked test split is unspent.** Nothing survived dev and validation, so
  there was nothing to take to it.

---

## 1. Where gold's return actually accrues

`drift_decomposition` splits a price series three ways: contiguous minutes
inside the roll hours (21:00–23:59 UTC), everything no contiguous minute covers
(the daily halt, weekends), and everything else. Only the third is reachable by
a strategy that goes home flat.

| symbol | split | years | total % | roll hrs | gaps | **liquid hrs** | **liquid %/yr** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **XAUUSD** | dev | 4.8 | +27.4 | +14.1 | +11.4 | **+2.0** | **+0.42** |
| **XAUUSD** | validation | 1.8 | +47.3 | +7.9 | +3.6 | **+35.8** | **+19.39** |
| USTEC | dev | 4.8 | +61.6 | +3.2 | +0.9 | +57.5 | +11.92 |
| USTEC | validation | 1.8 | +29.7 | +10.3 | +1.3 | +18.1 | +9.91 |
| EURUSD | dev | 4.9 | +0.1 | +9.1 | −1.9 | −7.1 | −1.46 |
| USDJPY | dev | 4.9 | +25.6 | −0.5 | −4.5 | +30.6 | +6.30 |

Read the first two rows against the third. Same corpus, same window, opposite
composition: **93% of gold's dev return sits in the roll and the gap, against
7% of USTEC's.** And gold's validation row flips — +35.8% of +47.3% in liquid
hours, which is the 2024–25 bull market and is genuinely gold moving.

So gold over this corpus is two different instruments: a financing vehicle in
2020–2023, and a trending asset in 2024–2025. Nothing in the data says which one
the next period will be.

### What the roll costs

| | overnight basis | annualised |
| --- | ---: | ---: |
| dev | +2.564 bps/night (t = 3.85) | **+6.67%/yr** |
| validation | +2.951 bps/night (t = 2.38) | **+7.67%/yr** |

The largest of the four instruments by a wide margin, and it points the wrong
way: gold yields nothing and costs financing, so a long **pays**. Charging it:

| swap, bps/night | dev CAGR | validation CAGR |
| --- | ---: | ---: |
| 0.00 (as usually reported) | +4.48% | +33.69% |
| 2.00 | −2.58% | +24.49% |
| **2.56 (the measured basis)** | **−4.47%** | **+22.03%** |
| 3.00 | −5.93% | +20.13% |

Any gold backtest run at zero swap that holds overnight is reading a number
about seven percentage points a year too kind.

---

## 2. The rejection log

| # | hypothesis | result | verdict |
| --- | --- | --- | --- |
| 1 | Drift available to an intraday strategy | **+0.42%/yr** over dev; +19.4%/yr over validation | one regime, not an edge |
| 2 | Hour-of-day drift shape | dev vs validation correlation +0.22 (60m), **−0.02** (30m) | noise |
| 3 | Intraday window selection | dev SR −0.07 to −0.68, validation −0.68 to −1.12 | negative in sample too |
| 4 | Minute-level autoregression | best cell **0.28x** of a round turn | too small |
| 5 | LBMA AM auction, 10:30 London | real at the mid (+0.98 bps, t 2.68); **+0.02 bps at real fills** | the effect is the spread |
| 6 | LBMA PM auction, 15:00 London | dev t −1.51 net, validation −1.47; no window clears | no |
| 7 | Daily momentum, 5–250 day | **every horizon negative in dev, every horizon positive in validation** | sign flips wholesale |
| 8 | Daily reversal | dev −0.81, validation +1.87 | sign flips |
| 9 | Volatility regime → return | dev −0.88 / +4.38 / +1.71, validation +21.04 / +4.69 / +13.49 | validation is just the bull |
| 10 | Risk-managed long overlay | trend gate **loses 4.65%/yr in dev**; inert in validation | actively harmful |
| 11 | Safe-haven conditioning on equity stress | risk-off avg weight 0.26, CAGR +1.28% dev; weak and thin | not established |
| — | Session opening-range breakout | evaluated in an earlier study and rejected there | out of scope by request |

### The two worth reading

**#5, the LBMA fix, was the best candidate in the whole project.** It is the only
cross-split-consistent directional signal found on any of the four instruments
that is not the carry roll, and it has a real mechanism: a scheduled, published,
one-sided auction. Gold drifts down through it.

| 10:15 → 10:45 London, short | n | mid, no cost | at real bid/ask + commission |
| --- | ---: | ---: | ---: |
| dev | 1,011 | **+0.981 bps** (t 2.68) | **+0.022 bps** (t 0.06) |
| validation | 386 | +0.571 bps (t 1.09) | +0.121 bps (t 0.23) |

The entire signal is the spread. Measured at the mid it looks like a 2.5%/yr
strategy; measured at the prices a trade actually gets, it is two hundredths of
a basis point. And year by year the dev result is 2020 on its own:

```
2020  +1.941 bps   2021  -0.750   2022  -0.389
2023  -0.589       2024  -0.272   2025  +0.924
```

Three window variants were tried around the auction and none clears. Trying more
would be fishing — the mid-level effect is 0.98 bps and the round turn is 1.05,
so no window can clear it unless the effect grows, and it does not.

**#10, the overlay, does not merely fail — it destroys value.** Pointing the
USTEC configuration at gold:

| dev | CAGR | vol | Sharpe | max DD | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy and hold | +4.48% | 15.3% | 0.36 | −23.2% | — |
| trend gate only | **−4.65%** | 9.4% | **−0.46** | −27.0% | 14.7x |
| gate + vol target | **−5.37%** | 11.0% | **−0.45** | −29.8% | 22.2x |

A −9.1 percentage point a year swing against simply holding, and the drawdown
gets *worse*. This is not a cost problem — 14.7x turnover at 1.05 bps is 7 bps a
year. It is a timing problem: gold spent 2020–2023 chopping across its own
200-session average, so the gate was out for the rebounds and in for the falls.
In validation the gate is inert, because gold never went below its average.

A rule that loses 9 points a year in one regime and does nothing in the other is
not a candidate, and it was not taken to the test split.

---

## 3. Why gold looked like the best remaining bet, and why that was wrong

It was recommended at the end of the USTEC study on two grounds. Both turned out
to be right about the surface and wrong about what mattered.

**"Gold has a real drift like USTEC."** It has a real *price* drift. The USTEC
study worked because there was a positive-drift asset to hold and the job was
sizing it — and USTEC's drift is in liquid hours, so it survives being held.
Gold's dev drift was financing, which does not survive the charge that offsets
it.

**"Gold runs near 15% volatility, so the sizing rule would shrink rather than
lever"** — the failure mode that sank USDJPY. That was correct: gold's average
weight came out at 0.52, neither levered nor squeezed. It did not help, because
the problem was never the sizing. It was the gate, and the gate is only useful
on an instrument that trends.

The general lesson, now visible across four instruments: **a volatility-sizing
overlay needs its host to have a drift in the hours you can hold it, and a trend
worth gating.** USTEC has both. Gold has neither in dev, and both in validation
— which is the same as saying gold had one good regime.

---

## 4. What was kept

`qlab.rollover.drift_decomposition(symbol, split=...)` and `drift_table()`.

Stage 2 listed overnight swap under "not modelled" and the EURUSD study measured
the price half of it. This answers the question that actually decides a
deployment: **how much of what this instrument returned is that adjustment?** It
is the sharpest instrument-selection tool in the project, and it took about
twenty lines.

Two of its numbers are asserted in `tests/test_rollover.py` rather than only
described, so that a future change which contradicts them breaks the build:
gold's dev liquid drift is under 1%/yr with a financing share above 85%, and
USTEC's is above 8%/yr with a financing share under 15%.

---

## 5. What would change the answer

- **A live swap reading, first.** Gold's basis is +2.56 bps a night. If a
  particular account charges materially less than that on a long, the arithmetic
  in §1 moves in gold's favour and buy-and-hold becomes worth revisiting. If it
  charges more — which is the way to bet — gold is a worse hold than the zero-
  swap numbers everywhere suggest. One line of MQL5.
- **More data, for the regime question.** The honest reading of §1 is that this
  corpus contains one financing regime and one bull regime, and 6.5 years is not
  enough to say which is normal. That is not fixable by searching harder.
- **Not the fix, and not the overlay.** The auction effect is the size of the
  spread and cannot outgrow it; the overlay is negative in the only regime that
  tested it.

**Gold's test split is unspent**, which is the one thing this study leaves in
better shape than the USDJPY one. A future gold hypothesis still has a clean
final estimate available — provided it is a genuinely different hypothesis and
not another configuration of the ten above.
