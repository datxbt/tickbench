# Small-cap retail strategies - evaluation

Subject: Poudel (2025), *Small-Cap Stock Trading Strategies for Retail Traders*
(SSRN 5921742). Six strategy families on US equities with market cap $300M-$2B.
The paper reports out-of-sample Sharpe ratios of **0.82** for family A
(volatility-scaled momentum) and **0.88** for family F (regime-filtered
composite) over 2022-2024, against 0.45 for IWM, net of $0.02-$0.05 spreads,
$0.01 slippage a side and zero commission.

Method: families A, D and F implemented in
[`src/qlab/strategies/smallcap.py`](../../src/qlab/strategies/smallcap.py) and
run by [`scripts/backtests/backtest_smallcap.py`](../../scripts/backtests/backtest_smallcap.py) on a
**point-in-time S&P SmallCap 600 panel** rebuilt from the index change history -
868 tickers, 1.74M ticker-days, 2018-2026. Exits are swept on the paper's own
in-sample window and carried unchanged to its out-of-sample window. Costs are the
paper's own numbers.

**Verdict: not deployable, and the reason is arithmetic rather than opinion. The
entry signals are real - family A's cost-free out-of-sample Sharpe is +1.19 and
its cost-free annual return +33.6% - but the strategy turns a $18-35 stock over
every three days, and the paper's own round turn of $0.07 a share costs 98-114%
of the entire gross edge. Net of that cost, not one of the 120 exit
configurations swept in-sample is profitable, out-of-sample family A returns
-0.6% a year at a Sharpe of +0.09 (t = 0.16, p = 0.88), and the composite family
F is significantly *worse* than simply holding IWM (Sharpe difference -0.84,
p = 0.049). Separately, the paper's own results table cannot be reconciled with
market data: it reports IWM returning +18.1% cumulatively over 2022-2024, when
IWM actually returned +2.2%. The post-publication holdout was not spent.**

---

## 0. What this corpus can and cannot say

**The universe is a proxy, and a defensible one.** The paper names no index and
ships no constituent list; it defines the universe by market cap and dollar
volume. The S&P SmallCap 600 is the closest published set - it is the $300M-$2B
band by construction - and unlike the Russell 2000 its **full change history** is
published, which is what makes a point-in-time reconstruction possible. Daily
membership is rebuilt by walking that history backwards from today, exact from
2019-12-17; before that date membership is frozen at its 2019-12 value, which
affects only the in-sample period.

**Survivorship bias is measured, not assumed away.** Of the 1,079 tickers that
were in the index at some point in the window, **211 (19.6%) no longer resolve** -
they were acquired, delisted or went to zero. Those are exactly the names that
performed worst. The panel therefore *flatters* the strategy, which matters for
reading the verdict: a strategy that fails on a survivor-biased panel fails
harder on the real one.

**Three of six families are out of reach.** B (opening-range breakout) and C
(intraday mean reversion) are specified on minute bars, and no free source
carries minute equity data back to 2018. E needs a point-in-time earnings
calendar. What stands in: this project has already tested the opening-range
geometry on its own tick corpus - see
[opening-range-breakout.md](opening-range-breakout.md), where the edge exists on
dev and does not carry to validation. Family F is implemented as *flat when
risk-off*, because its risk-off branch is family C; that is the charitable
reading, since the alternative leaves family A running into the downtrend.

**The splits are the paper's own, plus one it could not have seen.**

| split | period | use |
| --- | --- | --- |
| `is` | 2018-01-01 - 2021-12-31 | the paper's in-sample; the entire exit sweep lives here |
| `oos` | 2022-01-01 - 2024-12-31 | the paper's out-of-sample; the headline claim |
| `post` | 2025-01-01 - 2026-08-31 | after the paper was written - **unspent** |

Nothing is selected on `oos`. `post` stays closed because nothing survived `oos`
to justify opening it.

## 1. The paper's results table does not match the market

The strategy numbers cannot be checked directly. The benchmark numbers can, and
three of the six are wrong by margins no data vendor explains.

| benchmark, OOS 2022-2024 | paper | actual | gap |
| --- | --- | --- | --- |
| SPY Sharpe | 0.58 | 0.56 | ok |
| SPY annual return | 8.2% | 8.7% | ok |
| SPY max drawdown | -35.2% | **-24.5%** | -35.2% is the 2020 crash, not this window |
| IWM Sharpe | 0.45 | **0.15** | 3x |
| IWM annual return | 6.1% | **0.7%** | 5.4 pp |
| IWM cumulative return | +18.1% | **+2.2%** | 8x |

IWM returned -20.5% in 2022, +16.8% in 2023 and +11.4% in 2024. There is no
compounding of those that reaches +18.1%.

The table is also internally inconsistent. Section 7.3 lists family F's
out-of-sample years as **-15.1%, +21.4%, +9.8%**, which compound to **+13.2%**.
Two sentences later the same section states a cumulative out-of-sample return of
**+45.2%**, and Table 7.2 states an annual return of 13.9%, which compounds to
+47.8% over three years. The year list and the headline cannot both be right.

None of this proves the strategy numbers were not measured. It does mean the
results chapter was not produced from the 2022-2024 tape, and that is the only
evidence the paper offers for the 0.88.

## 2. The entry signals are real

Before costs, family A does what the paper says it does.

| family | split | cost-free Sharpe | cost-free annual return | trades |
| --- | --- | --- | --- | --- |
| A momentum | is 2018-21 | +0.63 | +10.4% | 2,051 |
| A momentum | oos 2022-24 | **+1.19** | **+33.6%** | 1,558 |
| D breakout-retest | is | +0.19 | +1.6% | 829 |
| D breakout-retest | oos | -0.46 | -7.2% | 626 |
| F composite | is | +0.39 | +5.0% | 1,679 |
| F composite | oos | +0.35 | +4.8% | 1,059 |

Family A's out-of-sample gross Sharpe of +1.19 is *higher* than the 0.82 the
paper claims net. Short-term momentum with a liquidity screen is a real effect in
small caps over 2022-2024, and the paper is not wrong about that. Everything that
follows is about what it costs to harvest.

## 3. The cost is the entire edge

The paper's own assumptions - $0.05 spread crossed once, $0.01 slippage each
side, $0 commission - come to **$0.07 a share** per round turn. On the stocks
these families actually select that is 20-38 basis points, and the families
trade every few days.

| family, split | median entry | round turn | median hold | gross $/trade | cost $/trade | net $/trade | cost as share of gross |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A, is | $18.61 | 37.6 bps | 3 days | +$4.06 | $4.60 | **-$0.55** | **114%** |
| A, oos | $21.13 | 33.1 bps | 3 days | +$7.28 | $7.15 | +$0.13 | **98%** |
| D, is | $31.44 | 22.3 bps | 9 days | +$3.28 | $4.08 | -$0.80 | 125% |
| D, oos | $35.74 | 19.6 bps | 9 days | -$4.86 | $3.81 | -$8.67 | - |
| F, is | $19.36 | 36.1 bps | 3 days | +$1.63 | $4.14 | -$2.51 | 253% |
| F, oos | $21.93 | 31.9 bps | 4 days | -$1.78 | $3.27 | -$5.05 | - |

Family A out-of-sample is the best case in the whole study, and it clears its own
transaction costs by **thirteen cents a trade**. That is not a margin anyone
deploys against; a one-cent error in the spread assumption reverses it.

The mechanism is the exit geometry. The sweep's best in-sample configuration is a
1-ATR target against a 3-ATR stop. That wins 69% of the time and still has a
profit factor of 1.00 - a high win rate paid for by rare losses three times the
size of the wins. The gross expectancy is near zero *by construction*, and a
33 bps round turn on a three-day hold is enough to put it below zero.

## 4. No exit rule anywhere on the grid works

Section 5.1.2 states three exit rules that partly contradict each other - a time
stop listed under "stop loss", and an intraday 2% rule that daily bars cannot
observe - so a verdict resting on one of them would be a verdict about the
choice. All 60 combinations of target (0.5-3.0 ATR), stop (1.0-3.0 ATR) and
horizon (5, 10, 20 days) were run on the in-sample period, for both families.

| family | configs | best Sharpe | median | worst | share profitable |
| --- | --- | --- | --- | --- | --- |
| A | 60 | -0.05 | -0.69 | -2.00 | **0 of 60** |
| D | 60 | -0.01 | -0.98 | -2.47 | **0 of 60** |

Not one configuration out of 120 has a positive net Sharpe on the paper's own
in-sample period, with the paper's own universe definition and the paper's own
costs. The verdict does not depend on which exit was picked, because there is no
exit that works.

## 5. Out-of-sample, against the benchmarks that matter

The in-sample winner of each sweep was carried unchanged into 2022-2024.

| strategy | net Sharpe | annual return | max DD | trades | Sharpe vs IWM | p |
| --- | --- | --- | --- | --- | --- | --- |
| A momentum | +0.09 | -0.6% | -25.3% | 1,558 | -0.05 | 0.92 |
| D breakout-retest | -0.58 | -8.2% | -26.2% | 626 | -0.73 | 0.057 |
| F composite | **-0.69** | -7.9% | -26.6% | 1,059 | **-0.84** | **0.049** |
| *paper's claim for F* | *0.88* | *13.9%* | *-16.1%* | - | - | - |
| IWM | +0.15 | +0.7% | -26.9% | - | - | - |
| SPY | +0.56 | +8.7% | -24.5% | - | - | - |
| equal-weight universe | +0.18 | +1.5% | -25.2% | - | - | - |

Family A's +0.09 carries a Lo standard error of 0.59: **t = 0.16, p = 0.88**, and
a stationary-block bootstrap interval of [-1.00, +1.23]. It is indistinguishable
from zero and indistinguishable from IWM (p = 0.92).

Family F - the paper's headline, claimed at 0.88 - is the *worst* of the three
here, and the only one whose shortfall against the benchmarks is significant:
-0.84 Sharpe against IWM (p = 0.049) and -0.85 against the equal-weighted
universe (p = 0.039). The regime filter does not rescue the composite; it cuts
average exposure to 25% while leaving the cost drag proportionally untouched.

## 6. The position sizing cannot produce the returns claimed

Section 5.1.3 sizes positions as `shares = R / (sigma * price)`, so a position's
*dollar* size is `R / sigma` and carries **no dependence on the account at all**.
The paper's own worked example is a $250 position. Measured across this panel,
ten concurrent slots under that formula come to **9-18% average gross exposure**;
the account is 82-91% in cash.

| family, split | exposure, paper sizing | exposure, volatility-target sizing |
| --- | --- | --- |
| A, oos | 12.2% | 61% |
| D, oos | 17.6% | 45% |
| F, oos | 9.2% | 25% |

A book that is 88% cash cannot return 12.4% a year unless the deployed eighth
returns over 100%. Every headline result above therefore uses a volatility-target
variant that keeps the paper's idea - size inversely with volatility - and drops
the accidental scale, so a full book is fully invested. Results under the paper's
literal formula are reported alongside in
[`smallcap_runs.parquet`](../../reports/strategies/smallcap_runs.parquet); they are not better.

## 7. What would have to be true for this to work

The gap is quantified, so the conditions are stated rather than guessed.

* **The round turn would have to fall by roughly an order of magnitude.** Family
  A out-of-sample earns 33 bps gross and pays 33 bps. Sub-penny spreads on
  $20 small caps with $500K of daily volume are not available to a retail
  account; they are not available to anyone crossing at market.
* **Or the holding period would have to lengthen by an order of magnitude.**
  A three-day hold paying 33 bps is 28 percentage points of annualised cost.
  The same signal held sixty days pays 1.4.
* **Or the exit geometry would have to stop being 1:3 against a 69% win rate.**
  The sweep says no point on the grid does this. The signal's own MFE/MAE
  structure, not the exit, is the binding constraint.

None of these is a tuning change. The first two are different strategies.

## 8. Reproducing

```bash
python scripts/pipeline/fetch_smallcap_universe.py
python scripts/backtests/backtest_smallcap.py
python scripts/research/analyse_papers.py --which smallcap
python -m pytest tests/test_smallcap.py
```

Artifacts: [`smallcap_runs.parquet`](../../reports/strategies/smallcap_runs.parquet) (every family x
sizing x cost x split), [`smallcap_sweep_is.parquet`](../../reports/strategies/smallcap_sweep_is.parquet)
(the 120-point exit surface), [`smallcap_equity.parquet`](../../reports/strategies/smallcap_equity.parquet)
(daily curves next to the benchmarks), `smallcap_trades_*.parquet` (trade tapes),
and [`paper_replication_analysis.json`](../../reports/strategies/paper_replication_analysis.json).

## 9. Verdict

**Not deployable.** The paper's entry signal is real and its cost assumption is
honest; the two are simply incompatible, and the paper never puts them side by
side. Out-of-sample the best family clears its own stated transaction costs by
thirteen cents a trade and returns -0.6% a year, the composite the paper
recommends is significantly worse than holding the small-cap index, and no exit
rule on a 120-point grid is profitable in-sample. The results table the claim
rests on reports benchmark returns that the market did not produce.

The post-publication period 2025-01 to 2026-08 was **not** spent. Nothing reached
it worth an unbiased estimate.
