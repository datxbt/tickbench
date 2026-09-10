# The engulfing candle - evaluation

Subject: the body-engulfing candle, stated exactly.

* **Long** — `close[1] < open[1]` and `close[0] > open[0]` and `open[0] <= close[1]` and `close[0] >= open[1]`
* **Short** — the exact mirror
* **Entry** — market, at the close of the signal candle
* **Stop** — `low[0]` for a long, `high[0]` for a short
* **Target** — 2R first, then swept for the most effective ratio
* **Size** — 0.01 lot
* **Exit** — stop or target, whichever comes first; on an opposite signal, close and reverse

Method: implemented in `src/qlab/strategies/engulfing.py` and run tick-by-tick
over 2020-01 to 2025-06 on all four instruments at **eleven timeframes from 1m
to 4h** — 1m, 2m, 3m, 5m, 10m, 15m, 20m, 30m, 1h, 2h, 4h. **1,827,321 signals; 1,576,016
trades taken at the stated 2R exit.** Fills cross a real bid and a real ask from the tape,
so spread is observed rather than assumed; commission is the published contract
term; slippage is `qlab.costs.CostModel` at the default 250 ms / 0.5 adverse
fraction, charged on every market leg — the entry, a stop-out, a reversal — and
never on a fixed target, which is a limit and fills at its price or not at all.

The idea nominates 2R only provisionally, so twelve exits are resolved against
the same fills: targets at 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5 and 6R, plus
the no-target variant that rides to the stop or the reversal. Everything is in
**R**, multiples of the planned risk from the candle's close to its own extreme
— the only unit in which a 1.3 bps EURUSD 1m stop and a 54 bps USTEC 4h stop are
the same bet.

> **Verdict: do not trade this. Rejected on dev, rejected again on validation.
> The held-out test split was not spent.**
>
> Across 528 (instrument × timeframe × exit) cells on dev, **zero** clear
> t = +2 and **435** sit at or below t = −2. Of the 44 (instrument × timeframe)
> cells at the stated 2R target, 43 lose on dev and 40 lose on validation.
>
> The loss is almost exactly the transaction cost, and the gross edge is
> **zero** — not small, not marginal: measured in money at 0.01 lot, adding
> every cost back leaves −$7,013 on dev against $111,469 of cost paid, and
> +$824 on validation against $40,462. The pattern's barriers resolve like a
> fair coin, and section 4 shows them landing on the coin's value to within
> half a percentage point.

---

## 1. The stop is the candle, so the cost is the timeframe

The entry is the candle's close and the stop is that same candle's extreme, so
**1R is one bar's range** — by construction, never more. 1R therefore shrinks
with the timeframe while the round turn does not.

Round-turn cost as a fraction of the amount risked, median over dev:

| | 1m | 5m | 15m | 30m | 1h | 4h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 0.420 | 0.198 | 0.115 | 0.080 | 0.057 | 0.025 |
| USDJPY | 0.523 | 0.244 | 0.142 | 0.100 | 0.069 | 0.031 |
| XAUUSD | 0.440 | 0.185 | 0.106 | 0.073 | 0.052 | 0.026 |
| USTEC | 0.367 | 0.156 | 0.089 | 0.061 | 0.043 | 0.020 |

On one-minute bars the median trade pays **37% to 52% of everything it risks**
just to get in and out. The median 1m risk is 1.3–2.9 bps of price; the round
turn on this account is 0.55–1.1 bps. Those are the same order of magnitude, and
that is the whole of the fast end of this study.

Note that this is *not* the pause-bar failure mode in disguise. The pause bar
selected for a small stop and was punished for it. The engulfing candle selects
for a **large** one — a body that swallows the previous body — and its cost/R is
roughly three-quarters of the pause bar's at the same timeframe. It is the better
setup of the two on precisely the dimension that killed the other one. It still
loses, and for a different reason.

## 2. The loss is the cost, to the dollar

Every trade's outcome, with the spread, the commission and the slippage added
back, is its cost-free equivalent. Summing that in money at 0.01 lot weights
each trade by what it actually risked, which is the honest way to aggregate when
1R varies fifty-fold across the table:

| dev, 44 cells, 1,124,991 trades at 2R | |
| --- | ---: |
| cost paid | $111,469 |
| net P&L | −$118,483 |
| **gross (cost added back)** | **−$7,013** |
| gross as a share of the cost paid | −6.3% |
| gross per trade | −$0.006 |

| validation, 44 cells, 451,025 trades at 2R | |
| --- | ---: |
| cost paid | $40,462 |
| net P&L | −$39,638 |
| **gross (cost added back)** | **+$824** |
| gross as a share of the cost paid | +2.0% |

**94% of the dev loss is transaction cost**, and the residual is a small
genuinely negative edge. On validation the residual flips sign and is smaller
still. Both are zero to within the noise of a million trades.

The same statement per cell: regressing mean net R on mean cost/R across the 44
dev cells gives

```
net_2R = -0.984 x cost_r - 0.015        r = -0.995
```

A slope of −1 and an intercept of zero is precisely what "no gross edge, pay the
cost" looks like. There is nothing else in the result to explain.

## 3. The pattern predicts nothing

Strip the barriers off entirely and measure the unconditional forward move from
the entry mid, signed by the direction the signal says to take. No stop, no
target, no cost. In basis points, because R is a different size on every row:

* Over 44 dev cells, the median cell's 5-bar forward return is **−0.069 bps**,
  and it is **negative in 30 of 44 cells**.
* Twelve cells are significantly *negative* at |t| > 2. Three are significantly
  positive. If the engulfing candle leans anywhere, it leans gently **against**
  the direction it is supposed to signal.
* Maximum favourable and maximum adverse excursion over the next 20 bars are the
  same size in every single cell — 2.18 vs 2.18 at 1m, 2.49 vs 2.42 at 4h.
  Price runs as far against the signal as it runs for it.

**Against the null.** The control keeps the entire geometry — market entry at a
bar's close, stop at that bar's own extreme, direction from the bar's own body —
and drops only the engulfment, drawing a size-matched deterministic sample of
non-signal bars. If the engulfment carried information, the setup would beat it:

| dev, mean over 44 cells | signal | control | paired t |
| --- | ---: | ---: | ---: |
| gross 2R (cost added back) | −0.0098 R | −0.0062 R | −0.74 |
| cost-free 2R at mid | +0.0075 R | +0.0147 R | −1.52 |

The setup does not beat its null. It **loses to it**, insignificantly, and beats
it in only **17 of 44 cells** — worse than a coin. The engulfment is decoration;
what is being measured is the stop-at-the-bar's-extreme structure, which every
candle in the sample has.

## 4. The barriers resolve like a fair coin

This is the cleanest result in the study, and it is worth stating on its own.

Under a driftless random walk, a 2R target with a 1R stop is reached first
exactly **one third** of the time. So restrict to trades that actually resolved
at a barrier — excluding the 18% that ended at a reversal, so the two-outcome
arithmetic is exact — and compare the realised hit rate with 33.3%, and with the
higher rate the costs demand:

| dev | n | cost/R | needs | actual | vs. fair coin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1m | 372,630 | 0.473 | 49.1% | 29.5% | −3.8 |
| 5m | 85,077 | 0.212 | 40.4% | 30.2% | −3.1 |
| 15m | 28,908 | 0.123 | 37.4% | 31.0% | −2.3 |
| 30m | 14,607 | 0.086 | 36.2% | 31.6% | −1.7 |
| 1h | 7,393 | 0.061 | 35.4% | 32.4% | −0.9 |
| 2h | 3,809 | 0.042 | 34.7% | 33.2% | −0.1 |
| 4h | 1,891 | 0.028 | 34.3% | **33.8%** | **+0.5** |

Validation reproduces it: 29.6% at 1m, 31.1% at 5m, 32.9% at 30m, 33.9% at 2h,
**34.0% at 4h**.

Read the last column. As the round turn shrinks toward nothing, the realised hit
rate converges on the fair-coin value from below and lands on it. **At 4h the
engulfing candle resolves its own stop and target within half a point of a
driftless random walk.** The shortfall at the fast end is not the pattern failing
— it is the spread pushing the entry away from the mid, which moves the target
further and the stop nearer by exactly the cost. Subtract the cost and there is a
coin underneath, at every timeframe.

## 5. No target rescues it

The full sweep, mean net R per trade averaged over all 44 dev cells, and how
many of those cells finish positive:

| exit | 0.5R | 1R | 1.5R | 2R | 3R | 4R | 5R | 6R | hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean R | −0.315 | −0.309 | −0.307 | −0.309 | −0.306 | −0.301 | −0.297 | −0.294 | −0.278 |
| positive cells | 0/44 | 1/44 | 1/44 | 1/44 | 4/44 | 5/44 | 5/44 | 5/44 | 7/44 |

The curve is **flat**. From 0.5R to 6R the mean moves by 0.02R while the cost
alone is 0.30R. There is no ratio that works, because the thing being optimised
is not there: you are choosing where to place a barrier on a random walk, and
every placement has the same expectation before costs and the same negative one
after. The mild improvement toward the right and toward `hold` is not an edge
either — it is fewer round turns per unit of time, i.e. paying the cost less
often.

The reversal rule is worth nothing measurable. Switching it off changes gross
from −6.3% of cost paid to −7.3%, and net from −$118,483 to −$86,749 on a third
fewer trades — the same rate of loss, applied less often.

## 6. The one place it is not negative, and why that is not the pattern

Twenty of the 528 cells are positive on dev *and* on validation. They deserve a
straight answer rather than a footnote, because they are the only thing standing
between this study and a uniform rejection.

They are not twenty findings. They are four instruments — XAUUSD 4h, USDJPY 4h,
USTEC 2h, EURUSD 4h — each appearing under several targets that share the same
underlying trades. Every one of them is a **slow, low-cost cell**, which is
exactly where a zero gross edge is no longer swamped by the round turn and can
land on either side of zero by chance. None clears t = +2 on dev; the best is
+1.65. Nothing was legitimately selected on dev, so nothing has earned a
validation confirmation, let alone the test split.

And the long/short split settles it. A symmetric pattern must pay symmetrically:

| validation, no-target exit | long | short |
| --- | ---: | ---: |
| 4h, pooled over four instruments | **+0.487 R** | −0.153 R |
| 2h, pooled over four instruments | **+0.284 R** | −0.053 R |
| XAUUSD 4h alone | **+0.860 R** | −0.215 R |

The entire profit is on the long side and the short side loses. Validation is
2024-01 to 2025-06 — gold roughly +60%, USTEC strongly higher. What the
no-target variant does on a 4h chart is hold a position for days and flip it on
the opposite signal, which in a bull market is a crude, mostly-long exposure.
That is drift capture wearing a candlestick costume, and it is available far
more cheaply and reliably from `qlab.strategies.risk_managed_long`.

## 7. A measurement trap worth recording

The obvious way to ask "does it work before costs?" is to price the same trades
at the mid. **That number is biased upward and must not be trusted here.**

A long's stop is triggered when the *bid* reaches it, so at the instant of a
stop-out the mid is already above the stop level. Measured over 300,000+ dev
stop exits, the exit mid sits **0.37 to 0.40 spreads favourably past the stop
level**, and the same applies to a target reached on the bid. The cost-free R
therefore inherits roughly half a spread of free money per trade — which is
largest precisely where the spread matters most, and produced apparently
positive "cost-free" numbers at 1m (+0.027 to +0.046 R) that are pure artifact.

The honest cost-free reads are the two used above: the **forward return from the
entry mid at a fixed horizon** (section 3), which involves no barrier timing at
all, and the **money decomposition** (section 2), which adds back the measured
cost rather than pretending it away. Both say zero.

The faded placebo exposes a second asymmetry worth recording: fading puts the
stop at the candle's *near* extreme, so 1R collapses by a factor of 4.3–5.0 and
cost/R explodes to 0.11–2.16. The setup's own stop placement is, structurally,
about five times better than its mirror's — the one genuine merit the geometry
has. It is not enough, because the thing it is protecting is worth nothing.

## 8. What would have to be true

For this to be tradable, one of these would have to hold, and none does:

1. **The pattern predicts direction.** It does not: forward returns are flat to
   slightly negative, excursions are symmetric, and it loses to a control that
   drops the engulfment entirely.
2. **Some exit ratio extracts value the signal has.** There is no value to
   extract; the curve from 0.5R to 6R is flat to within a tenth of the cost.
3. **A slow timeframe escapes the cost.** 4h does escape the cost — and lands
   exactly on the fair-coin hit rate, which is the same as having no edge.

## 9. Reproducing

```bash
python scripts/backtests/backtest_engulfing.py                       # 1m -> 4h, all four
python scripts/backtests/backtest_engulfing.py --control             # the geometry-only null
python scripts/backtests/backtest_engulfing.py --no-reverse          # what reversing is worth
python scripts/backtests/backtest_engulfing.py --placebo             # the faded signal
```

Artifacts: `reports/strategies/engulfing.json` (every table above as data) and
`reports/strategies/engulfing_trades/*.parquet` (the full trade tapes, one row
per signal per variant per split). Tests in `tests/test_engulfing.py` pin the
pattern's four-way conjunction, which side of the book each level acts on,
stop-before-target resolution, and the equivalence between the independently
priced exits and a literal stateful walk of the reversal chain.

**The test split (2025-07-01 onward) remains unspent.**
