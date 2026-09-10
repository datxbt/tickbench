# Decision trees for intraday signals - evaluation

Subject: Prajwal, Balivada, Nirmala and Poornoday (2024), *Decision Trees for
Intuitive Intraday Trading Strategies* (SSRN 4838381). Fit one depth-4 Gini
`DecisionTreeClassifier` per instrument on nine technical features of 1-minute
bars; label each bar by the **sign of the next bar's return**; shift the signal
forward one bar for execution delay; trade it. On NIFTY50 constituents they
report a **test-set Sharpe of 5.81** against buy-and-hold's 2.36, with a 2.63%
maximum drawdown against 7.02% and 48.15% of bars won.

Method: implemented in `src/qlab/strategies/decision_tree.py`. One tree per
instrument, fitted on the **dev** split (2020-2023, 1.3-1.4M bars each) and
measured on **validation** (2024 - H1 2025, ~520-550k bars each), on all four
instruments. Every result is reported twice: cost-free, which is what the paper
measured, and net of the measured cost model.

**Verdict: rejected, decisively and for a stated reason. The classifier has no
directional edge - held-out accuracy is within 0.03% of the majority-class base
rate on three of four instruments. The cost-free equity curve the paper's method
produces is not significant on any instrument (best t = 1.73), and it turns the
position over 27-134 times a day. The gross edge can afford a round turn of
0.037-0.080 bps; the round turn actually costs 0.58-0.98 bps, which is 7 to 22
times more. Net of costs all four instruments lose 27-91% inside 1.85 years. The
held-out test split was not spent.**

The paper states the cause itself, in Section IV.C: *"commissions and slippage
were not explicitly incorporated into the backtesting process."*

---

## 0. What this corpus can and cannot say

**The universe.** There is no NIFTY50 here. The paper's method is explicitly
per-name - *"decision trees create unique trading rules for each stock"* - so it
transfers to four instruments as four independent fits. Four is not fifty, and
their PSBBR / PSBBS statistics (share of names beating their own benchmark)
become a count out of four that moves 25 points at a time. What four instruments
*can* answer is whether the finding is instrument-specific or universal, and
here the answer is unambiguous in a way a larger universe would not improve.

**The split is stricter, not looser.** Their train/test boundary is a date they
chose after seeing the data. This project's splits were fixed in
`qlab.loader.SPLITS` before any strategy existed, so the tree trains on dev and
is measured on validation with no boundary chosen to suit it.

**Volume is quote updates.** Feature f9's VWAP needs volume and the feed carries
no size, so `n_ticks` stands in - what MT5 reports as volume on these
instruments.

**Annualisation.** The paper annualises 1-minute returns by 252 x 375 = 94,500
periods. That is arithmetically fine and makes their Sharpe uncomparable with
every other number in this project, so everything here aggregates net minute
returns into **daily** returns and annualises by 252, with their convention
reported alongside. On the same data the two differ by roughly a factor of two,
which is itself worth knowing when reading a 5.81.

**Spot CFDs are not equities.** An Indian equity has a closing auction, an
overnight gap and a 375-minute session; XAUUSD trades 24/5. The paper's method
does not depend on any of that - it reads nine indicators and predicts the next
bar - but the transfer is not free and is flagged rather than argued away.

## 1. The classifier does not predict direction

Held-out directional accuracy, against the majority-class base rate. Fifty
percent is **not** the bar: on a drifting instrument the up-bar share is not a
half, and a tree that learned only the drift would clear 50% while predicting one
constant.

| instrument | in-sample | held-out | base rate | edge over base | features used |
| --- | --- | --- | --- | --- | --- |
| XAUUSD | 50.800% | 50.412% | 50.415% | **-0.003%** | 5 of 9 |
| USTEC | 51.364% | 50.683% | 50.716% | **-0.033%** | 6 of 9 |
| EURUSD | 52.834% | 52.987% | 52.980% | **+0.007%** | 5 of 9 |
| USDJPY | 52.665% | 52.016% | 51.682% | **+0.334%** | 5 of 9 |

Three of the four are indistinguishable from predicting the majority class.
USDJPY's third of a percent is the only non-trivial number in the column, and
Section 3 shows what it is worth.

One prediction of the paper *does* replicate: it says most fits use five to
seven of the nine features and that the set differs by name. Measured: five,
six, five, five, and the sets differ - XAUUSD splits on `vwap_ratio` and no
other instrument does; USTEC is the only one using `ret15`. The trees are
genuinely idiosyncratic. They are also, on the evidence above, idiosyncratically
uninformative.

The fitted XAUUSD tree, for concreteness:

```
|--- rsi14 <= 39.220951
|   |--- ret1 <= -0.000286
|   |   |--- ret1 <= -0.000610
|   |   |   |--- rsi14 <= 16.544384 -> class 1
|   |   |   |--- rsi14 >  16.544384 -> class 1
...
```

Both children of the `rsi14 <= 16.54` split predict the same class. That is a
depth-4 tree spending a split to distinguish two cases it then treats
identically - the shape of a fit finding nothing.

## 2. The paper's own backtest, reproduced cost-free

Their execution rule, their cost assumption (none), on validation:

| instrument | Sharpe (daily, 252) | their annualisation | total % | max DD % | vol % | B&H Sharpe | B&H total % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XAUUSD | 1.02 | 0.58 | +17.6 | -5.5 | 9.01 | 1.82 | +60.1 |
| USTEC | 0.44 | 0.28 | +12.3 | -25.2 | 18.14 | 0.79 | +33.0 |
| EURUSD | 1.08 | 0.58 | +5.0 | -3.2 | 2.47 | 0.53 | +6.5 |
| USDJPY | 1.32 | 0.65 | +11.4 | -3.4 | 4.50 | 0.16 | +2.0 |
| *paper (NIFTY50)* | *5.81* | | *+28.6* | *-2.6* | *3.32* | *2.36* | *+34.2* |

Two features of this table are worth separating.

**The qualitative pattern reproduces.** Lower volatility than buy-and-hold on
every instrument, shallower drawdown on three of four, and a Sharpe that beats
the benchmark on two - which is exactly the shape of the paper's own finding
(their total return underperforms, their Sharpe outperforms). This is not
surprising and is not evidence: a strategy in the market 3.5% to 61% of the time
has lower volatility than one in it always, and dividing a similar return by a
smaller denominator raises a Sharpe. The paper's Sharpe advantage is largely an
exposure difference reported as a skill difference.

**None of it is significant.** Daily gross returns, bootstrapped:

| instrument | gross daily mean | 95% CI | p | Sharpe t | exposure |
| --- | --- | --- | --- | --- | --- |
| XAUUSD | +3.65 bps | [-0.92, +8.29] | 0.121 | 1.37 | 34.5% |
| USTEC | +3.18 bps | [-5.89, +11.89] | 0.476 | 0.63 | 61.0% |
| EURUSD | +1.06 bps | [-0.51, +2.61] | 0.184 | 1.18 | 3.5% |
| USDJPY | +2.35 bps | [-0.02, +4.58] | 0.052 | 1.73 | 13.1% |

Four instruments, best p = 0.052 on the one with the largest accuracy edge.
Nothing here clears a bar that four simultaneous tests should have to clear -
and this is before a single unit of cost.

## 3. The cost that was not charged

This is the whole study. "Breakeven" is the round-turn cost at which the gross
edge is exactly consumed - total gross return divided by total turnover.
"Actual" is what a round turn costs on this account, from the measured model
(realised spread, published commission, measured slippage at 250 ms).

| instrument | turnover/day | breakeven bps | actual bps | ratio | net Sharpe | net total % |
| --- | --- | --- | --- | --- | --- | --- |
| XAUUSD | 133.6 | 0.0546 | 0.6816 | **12.5x** | -10.90 | **-85.9%** |
| USTEC | 113.0 | 0.0562 | 0.9802 | **17.4x** | -7.04 | **-91.3%** |
| EURUSD | 26.6 | 0.0798 | 0.5791 | **7.3x** | -5.93 | **-26.8%** |
| USDJPY | 126.4 | 0.0372 | 0.8247 | **22.2x** | -18.77 | **-90.3%** |

A next-bar classifier on 1-minute data changes its mind roughly every six to
twenty minutes. The strategy needs to earn about a twentieth of a basis point per
round turn and the round turn costs most of one. It is not close, it is not
close on any instrument, and it is not a matter of a better broker: EURUSD on
raw-spread pricing with $5 round-turn commission is still seven times too
expensive.

This is why the verdict is decisive where the VWAP/EMA study's was merely
negative. That strategy had no edge and cost little. This one has no edge and
costs enormously, so even a real edge of the size the paper claims would not
survive: their reported +28.62% total return over the test period, at this
turnover, would need roughly 0.05 bps round-turn execution to exist.

## 4. Depth changes nothing

The paper's Figures 1-3 compare depths 3 to 6 and settle on 4. On XAUUSD:

| depth | held-out accuracy | gross Sharpe | net Sharpe | turnover/day | breakeven bps |
| --- | --- | --- | --- | --- | --- |
| 3 | 50.410% | 1.07 | -10.65 | 127.6 | 0.0589 |
| **4** | 50.412% | 1.02 | -10.90 | 133.6 | 0.0546 |
| 5 | 50.393% | 1.07 | -11.61 | 144.4 | 0.0533 |
| 6 | 50.382% | 0.92 | -11.79 | 150.2 | 0.0452 |

Accuracy varies by three thousandths of a percent across the whole range. What
depth actually does is monotonically raise turnover and lower the breakeven -
deeper trees are strictly worse net, for reasons that have nothing to do with
bias and variance. The paper's depth-4 choice is not wrong; there is simply no
tradeoff to be right about.

## 5. The execution rule, and what it reveals

The paper shifts the signal forward one bar. That means the model is trained to
predict bar `t+1`'s return and the position is then held over `t+2` - the
prediction and the position are one bar apart. On XAUUSD:

| lag | mapping | gross Sharpe | gross total % | net Sharpe | turnover/day |
| --- | --- | --- | --- | --- | --- |
| 0 | long/flat | 0.30 | +4.5 | -11.13 | 133.6 |
| 0 | long/short | -1.49 | -33.3 | -16.71 | 267.1 |
| **1** | **long/flat** | **1.02** | **+17.6** | -10.90 | 133.6 | *(the paper)* |
| 1 | long/short | -0.61 | -15.5 | -16.10 | 267.1 |

The paper's cost-free result is **three times better when the position is
applied to the bar the model does not predict** than when it is applied to the
bar it does. If the classifier had a directional edge, delaying it by a bar
would degrade it. That the reverse holds is direct evidence that the gross
"edge" in Section 2 is not prediction at all - it is a residual exposure
artefact, an accident of which bars a 34%-exposure long/flat rule happens to sit
out of during a rising sample.

The long/short mapping loses money gross on every configuration, which is the
same fact seen from the other side: with the flat leg removed, the strategy is
just a noisy long/short book on a coin flip.

## 6. Per-instrument spread (their PSBBR / PSBBS)

| | count | share | paper (50 names) |
| --- | --- | --- | --- |
| PSBBR, cost-free | 1/4 | 25% | 46% |
| PSBBS, cost-free | 2/4 | 50% | 60% |
| PSBBR, net of costs | 0/4 | **0%** | not reported |
| PSBBS, net of costs | 0/4 | **0%** | not reported |

The cost-free row is in the same neighbourhood as the paper's, on a sample far
too small to say more than that. The net rows are the ones that decide anything,
and the paper does not report them because it did not compute them.

## 7. What this does not say

- **It does not say decision trees cannot trade.** It says a depth-4 tree on
  these nine features, targeting the next 1-minute bar's sign, has no
  directional edge on four spot CFDs - and that at this horizon, the cost of
  acting on any edge is roughly twenty times the edge available.
- **It does not test Indian equities.** If NIFTY50 constituents at 1-minute
  resolution carry predictable next-bar direction that these instruments do not,
  this study cannot see it. The cost arithmetic in Section 3 would still apply:
  Indian retail equity costs (brokerage, STT, exchange fees, stamp duty) are
  well above 0.08 bps per round turn, so the paper's own market does not clear
  its own breakeven either.
- **It does not attempt to improve the model.** No feature engineering, no class
  weighting, no ensembling, no walk-forward refitting. The paper's method was
  run as specified.
- **The volume proxy is a proxy** (feature f9 only).
- **It is 1.85 years of held-out data** per instrument, though at 520k bars each
  the sample size is not the binding constraint on anything above.

## 8. If you want to keep pulling this thread

The one result worth a second look is that **exposure differences masquerade as
skill**. Every instrument shows lower volatility and shallower drawdown than
buy-and-hold, and two show a higher Sharpe, with a classifier that provably
predicts nothing. Any study that compares a partially-invested rule to a
fully-invested benchmark on Sharpe alone will find this, and the paper's Table 1
is exactly that comparison. A benchmark matched on average exposure - buy-and-
hold scaled to 34.5% of notional - would remove the effect, and is the control
that should accompany any result of this shape.

The breakeven-cost calculation in Section 3 is also the cheapest possible screen
for this entire class of strategy, and costs one line: divide the gross return by
the turnover before doing anything else. Applied first, it would have ended this
study in a minute rather than a morning.

## 9. Reproducing

```bash
python scripts/backtests/backtest_decision_tree.py
```

Writes `reports/strategies/decision_tree.json`. Roughly two minutes for all four
instruments. Tests: `pytest tests/test_decision_tree.py` (38 cases, including a
lookahead probe that perturbs a single future bar and asserts no feature at any
earlier bar moves, an ADX regression test for the 0/0 that once poisoned every
value through the Wilder smoother, and the full lag chain from prediction to
held position).

The `test` split was **not** read.
