# Machine-learned FX trading signals - evaluation

Subject: Enkhbayar and Slepaczuk (2024), *Predictive Modeling of Foreign Exchange
Trading Signals Using Machine Learning Techniques* (SSRN 4862571). Eight
regressors - Ridge, KNN, RF, XGBoost, GBDT, ANN, LSTM, GRU - predict the next
bar's return from technical indicators; the prediction is thresholded into
buy / hold / sell against the training set's quartiles; the position is held until
the signal changes. Everything is refitted in a rolling walk-forward (600/156/126
bars daily, 3600/936/756 on 4-hour data). An EMA-crossover trend follower, itself
walk-forward optimised over ten fast/slow pairs, is the traditional comparison,
and buy-and-hold is the benchmark. Cost is a fixed 0.02%.

Method: implemented in
[`src/qlab/strategies/fx_ml.py`](../../src/qlab/strategies/fx_ml.py) and run by
[`scripts/backtests/backtest_fx_ml.py`](../../scripts/backtests/backtest_fx_ml.py) on this project's
corpus - EURUSD, USDJPY, XAUUSD and USTEC, daily and 4-hour bars built from
tick-level data. **792 runs**: 4 instruments x 2 frequencies x 8 models x 3 signal
modes x 2 threshold rules x 2 feature encodings, plus the trend follower.

**Verdict: not deployable, and the paper broadly agrees. Across all 792 runs, the
median annualised Sharpe is -0.21 against +0.97 for holding the instrument. Under
a paired stationary-block bootstrap against buy-and-hold on the same bars, 156
runs differ from the benchmark at a Benjamini-Hochberg-corrected 5% level and
every single one of them is on the losing side - zero runs beat buy-and-hold at
any significance level, raw or corrected. The paper's own headline finding, "at
least one model beats the benchmark in each case", reproduces in 5 of 8 cases and
is a max-of-24 artefact: the best of each set has a paired p-value no lower than
0.27. Two specification choices are identified as sufficient causes and are
measured, not speculated about.**

---

## 0. What this corpus can and cannot say

**The sample barely overlaps the paper's.** They ran 2000-2023 on six pairs from
an ICMarkets feed. This corpus is 2020-2026 on an Exness feed, and the
walk-forward's first evaluable bar lands in mid-2022 because each window needs
882 daily or 5,292 four-hour bars of history before it can score anything. So
almost everything measured here is **out of sample for the paper**, which is the
useful thing a replication on a different corpus can be.

**Three FX pairs, not six, plus an index CFD.** EURUSD is common to both studies.
USDJPY, XAUUSD and USTEC are not in the paper; XAUUSD and USTEC are carried
because the method claims nothing FX-specific and a trending instrument is the
hardest test for a signal that competes with buy-and-hold.

**Five to seven walk-forward windows per instrument-frequency**, against the
paper's forty. The paper's window geometry is stated in absolute bar counts, so a
shorter corpus yields fewer windows rather than shorter ones - the honest failure
mode, and it is why single-run results here carry wide intervals and the verdict
rests on the distribution across runs rather than on any one of them.

| instrument | freq | evaluated bars | windows | period |
| --- | --- | --- | --- | --- |
| EURUSD | 1d | 756 | 6 | 2022-08 - 2025-03 |
| EURUSD | 4h | 3,780 | 5 | 2022-11 - 2025-04 |
| USDJPY | 1d | 882 | 7 | 2022-07 - 2025-06 |
| USDJPY | 4h | 3,780 | 5 | 2022-11 - 2025-04 |
| XAUUSD | 1d | 882 | 7 | 2022-07 - 2025-06 |
| XAUUSD | 4h | 3,780 | 5 | 2022-11 - 2025-04 |
| USTEC | 1d | 756 | 6 | 2022-08 - 2025-03 |
| USTEC | 4h | 3,780 | 5 | 2022-12 - 2025-05 |

**The locked test split was not touched.** The rolling walk-forward would
otherwise have marched straight through 2025-07 onwards on its own; the driver
stops at the end of `validation` unless `--allow-test` is passed, and it was not.

**Hyperparameter selection is explicit, not cross-validated.** Section 3.6
describes fitting on a training block and tuning on a separate validation block,
which is what happens here - every grid point is fitted on train and the lowest
validation MAE wins. The grids are narrower than the paper's RandomizedSearchCV
ranges; on 600 rows a wider grid buys variance, not accuracy.

## 1. EURUSD daily reproduces the paper's own table

The paper's Table 10 is the one directly comparable artefact. Their picture:
most models negative, the neural networks least bad, the trend follower at 0.17,
buy-and-hold at 0.22. Here, with their exact settings:

| model | buy & sell | only buy | only sell |
| --- | --- | --- | --- |
| ridge | -0.70 | +0.48 | -0.02 |
| knn | *never trades* | +0.70 | +0.15 |
| rf | -0.13 | +0.31 | -0.26 |
| xgboost | -0.39 | +0.54 | +0.05 |
| gbdt | +0.48 | -0.15 | -0.79 |
| ann | -0.39 | +0.07 | -0.53 |
| lstm | +0.29 | +0.37 | -0.17 |
| gru | -0.35 | +0.04 | -0.49 |
| **EMA cross (TF)** | **-0.21** | **+0.02** | **-0.08** |
| **buy & hold** | **+0.41** | **+0.41** | **+0.41** |

The qualitative shape matches: scattered around zero, mostly below the benchmark,
and the trend follower unimpressive - the paper's phrase is that moving-average
crossovers "are not efficient at all", and that reproduces exactly. This is a
successful replication of a negative result on a sample the authors never saw.

## 2. The only test that matters: paired against holding the instrument

Seven of the eight best runs in this study are `only_buy` - a mode that is long
whenever the model is not flat. On a rising instrument that is a *diluted
buy-and-hold*, and a Sharpe measured against zero rewards it for the asset's
drift. So every run is also compared with holding the instrument over exactly the
same bars, by stationary-block bootstrap on the paired difference.

| | all 792 runs |
| --- | --- |
| median Sharpe difference vs buy-and-hold | **-0.99** |
| runs with a positive point estimate | 78 (9.8%) |
| runs significantly different from B&H, raw p < 0.05 | 274 |
| ...of which **better** than B&H | **0** |
| runs significantly different after Benjamini-Hochberg | 156 |
| ...of which **better** than B&H | **0** |

Two hundred and seventy-four runs are statistically distinguishable from simply
holding the instrument, and all 274 are worse. Not one run in 792 beats
buy-and-hold at even an uncorrected 5% level. The best point estimate in the
whole study - Ridge on EURUSD 4-hour, +0.74 Sharpe over B&H - has an interval of
[-0.64, +2.08] and p = 0.30.

## 3. The paper's headline claim is a maximum, not an edge

The paper's central positive result is that "at least one individual
machine-learning based strategy outperforms the benchmark in each case". That
reproduces in 5 of 8 cases here - and means nothing, because each case takes the
maximum of 24 runs.

| instrument | freq | B&H | best ML | which | how many of 24 beat B&H | paired p |
| --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 1d | +0.41 | +0.70 | knn / only buy | 4 | 0.47 |
| EURUSD | 4h | +0.29 | +1.03 | ridge / buy & sell | 1 | 0.30 |
| USDJPY | 1d | +0.30 | +0.78 | xgboost / only buy | 4 | 0.31 |
| USDJPY | 4h | +0.29 | +0.76 | gbdt / only buy | 6 | 0.27 |
| USTEC | 1d | +1.06 | +0.85 | knn / only buy | **0** | 0.57 |
| USTEC | 4h | +1.40 | +1.59 | gbdt / only buy | 1 | 0.75 |
| XAUUSD | 1d | +1.39 | +1.25 | ridge / only buy | **0** | 0.71 |
| XAUUSD | 4h | +1.93 | +1.22 | ann / only buy | **0** | 0.10 |

Across all faithful ML runs, **8.4% beat buy-and-hold**. If the signals carried no
information relative to the benchmark, roughly half should - so the models are not
merely uninformative, they are actively worse than the thing they are trying to
improve on, which is what a cost applied to a coin flip looks like. And the two
strongly trending instruments, gold and the Nasdaq, are where nothing beats
holding at all.

## 4. Two specification choices are sufficient to explain the failure

Both are switchable in the module, so the failure can be attributed rather than
merely reproduced.

### 4.1 The gate is calibrated against the wrong distribution

Section 3.7 compares the *predicted* return to the first and third quartile of
the *training set*. But a regressor fitted with MAE loss on a near-unpredictable
target shrinks hard toward the mean, so its predictions live in a band far
narrower than the realised returns whose quartiles are the gate.

Measured shrinkage, `sd(prediction) / sd(target)`, by model:

| ridge | knn | xgboost | gbdt | rf | lstm | gru | ann |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.14 | 0.14 | 0.16 | 0.19 | 0.20 | 1.47 | 2.39 | 3.06 |

The five tree and linear models predict with a seventh to a fifth of the target's
spread. Their predictions therefore almost never reach a quartile of the realised
distribution, and under the paper's own gate the median run is **in the market
only 17-19% of the time** - KNN on EURUSD daily never opens a position at all. The
neural networks have the opposite problem: they emit predictions two to three
times wilder than anything that happens, so their gate is always open.

Neither is a signal. Switching the gate to the quartiles of the *predictions*
fixes the trading frequency and does not fix the returns - the median run is
still below buy-and-hold.

### 4.2 The features are price levels, and a tree cannot extrapolate

Table 3 lists EMA10-200, MACD, Bollinger bands and average price as features, in
levels. A tree fitted on 600 days of EURUSD between 1.05 and 1.12 and then asked
about 1.18 predicts from the boundary of its training box; every window's model
spends its test period extrapolating off the edge of what it saw.

Rescaling every price-scaled feature by that bar's close - leaving the bounded
oscillators alone - is the whole fix, and it is measured here as the `stationary`
feature mode. It changes which models look best. It does not produce a run that
beats buy-and-hold.

## 5. Cost, frequency, and the paper's research questions

The paper's RQ1 concludes that intraday data produces higher Sharpe ratios than
daily. On this corpus the opposite holds, and the mechanism is measurable.

| | daily | 4-hour |
| --- | --- | --- |
| median ML Sharpe (faithful settings) | -0.03 | **-0.48** |
| median trades per year | 37 | 245 |
| median cost drag, gross Sharpe - net Sharpe | 0.12 | **0.70** |

At 0.02% a turn, six-times-a-day bars cost six times as much per year, and 0.70 of
Sharpe is more than any of these models produces gross. The paper's own numbers
show the same thing in dollars - its 4-hour tables carry $1,400-2,800 of
transaction cost against $17,000 of net profit on a $1,000 account - but it reads
the higher *net profit* of intraday trading as evidence for intraday, when the
risk-adjusted comparison goes the other way.

RQ2 concludes that only-sell signals had the highest Sharpe. Here only-sell is
the **worst** mode and only-buy the best:

| signal mode | median Sharpe |
| --- | --- |
| only buy | **+0.38** |
| buy & sell | -0.23 |
| only sell | **-0.91** |

That reversal is the tell. Their sample ran through long stretches of dollar
strength, so a short bias in `USD/X` terms was a directional bet that paid; this
sample has rising gold, a rising Nasdaq and a weaker dollar, so a long bias pays.
Neither result is a property of the models. It is the drift of the sample,
arriving through whichever one-sided mode happens to face it.

RH3 concludes neural networks beat non-neural models. Here they do not: median
Sharpe -0.37 for ANN/LSTM/GRU against -0.15 for the rest.

## 6. Reproducing

```bash
python scripts/pipeline/prepare_fx_bars.py
python scripts/backtests/backtest_fx_ml.py
python scripts/research/analyse_papers.py --which fx
python -m pytest tests/test_fx_ml.py
```

The full run is 792 fits and takes about 35 minutes on CPU. Artifacts:
[`fx_ml_runs_devval.parquet`](../../reports/strategies/fx_ml_runs_devval.parquet) (one row per run),
[`fx_ml_returns_devval.parquet`](../../reports/strategies/fx_ml_returns_devval.parquet) (per-bar net
returns and positions),
[`fx_ml_paired_devval.parquet`](../../reports/strategies/fx_ml_paired_devval.parquet) (the paired
bootstrap against buy-and-hold),
[`fx_ml_analysis_devval.parquet`](../../reports/strategies/fx_ml_analysis_devval.parquet), and
[`paper_replication_analysis.json`](../../reports/strategies/paper_replication_analysis.json).

## 7. Verdict

**Not deployable**, and the paper's own conclusions point the same way - it
rejects two of its three research hypotheses and reports that moving-average
crossovers "are not efficient at all". What this replication adds is that the
positive fragment the paper does keep, "at least one model beats the benchmark in
each case", does not survive being tested properly: on a corpus the authors never
saw, across 792 runs, zero beat buy-and-hold at any significance level, and 156
are significantly worse after correcting for multiplicity.

The two identified causes - a gate calibrated against realised rather than
predicted quartiles, and price-level features a tree cannot extrapolate from -
are both fixable, and fixing them does not produce an edge. That is the useful
finding: the specification is not what is wrong. On four liquid instruments at
daily and 4-hour frequency, this feature set does not carry directional
information worth 2 basis points a turn.
