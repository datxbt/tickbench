# tickbench

Tick-data research platform for systematic strategy development on Exness
raw-spread feeds (EURUSD, USDJPY, XAUUSD, USTEC), 2020-2026.

The installed distribution is `tickbench`; the import name is `qlab`.

Thirty hypotheses - price-action folklore, published papers, machine-learned
signals - implemented against 696M ticks and priced with a cost model measured
from the same feed. **One survived as a deployable strategy, one more as a
forecasting model.** Eight more are open questions this data cannot settle. The
rest are written up in full, with the null they were tested against.

- **[docs/findings/](docs/findings/README.md)** - what was tested and what
  survived. Start here.
- **[docs/data-layer.md](docs/data-layer.md)** - storage format, cleaning
  policy, point-in-time convention, splits.
- **[docs/cost-model.md](docs/cost-model.md)** - spread, commission, slippage,
  and the turnover budget they imply.
- **[docs/data-quality.md](docs/data-quality.md)** - what the audit found, and
  the broker contract terms on file.
- **[docs/commands.md](docs/commands.md)** - every entry point in `scripts/`.
- **[docs/using-the-library.md](docs/using-the-library.md)** - reading data and
  charging cost from Python.

## Data

Exness Raw Spread tick history: every bid/ask quote, millisecond timestamps, no
trade size. 48 GB of vendor CSV, stored as 4.6 GB of monthly zstd parquet.

| symbol | ticks | first day (UTC) | last day (UTC) |
| --- | ---: | --- | --- |
| XAUUSD | 283,059,760 | 2020-01-29 | 2026-08-31 |
| USTEC | 216,564,864 | 2020-01-29 | 2026-09-01 |
| USDJPY | 109,419,415 | 2020-01-29 | 2026-08-31 |
| EURUSD | 86,878,906 | 2020-01-29 | 2026-08-31 |
| **total** | **695,922,945** | | |

Derived from it: 9.4M one-minute bars (right-edge labelled, so a bar is known at
its `ts`), and hourly spread and latency profiles per split for the cost model.
There is no volume - `n_ticks`, quote updates per bar, stands in for it.

**Splits**, fixed on 2026-09-02 before any strategy work and not moved since:

| split | dates | use |
| --- | --- | --- |
| dev | 2020-01-29 to 2023-12-31 | hypothesis generation, parameter search |
| validation | 2024-01-01 to 2025-06-30 | choosing between candidates that survived dev |
| test | 2025-07-01 to 2026-09-01 | locked; one run, at the end |

**Known breaks inside the data** - a strategy fitted across one of these is
fitted to two different markets:
- **Q4 2023:** the broker rebuilt its spread distribution. Minute-scale reversal
  structure that is real in 2020-2023 is gone afterwards.
- **June 2023:** USTEC's daily halt moved from 16:00-18:30 to 17:00-18:00 New
  York, which ended a t = 4 gap effect.
- **XAUUSD:** a financing vehicle with no holdable intraday drift in 2020-2023, a
  trending asset in 2024-2025, and bracket widths four to five times larger in
  2026.

**External data**, for the studies that need other markets: Binance BTCUSDT
5-minute klines 2020-2026 (569,490 bars), and a point-in-time S&P SmallCap 600
panel 2018-2026 reconstructed from the index change history (868 tickers, 1.74M
ticker-days).

**None of it is committed.** The tick data is vendor data and too large for git;
put the CSVs in `Tick_Data/` and rebuild everything else (see Setup). Details:
[docs/data-layer.md](docs/data-layer.md), [docs/data-quality.md](docs/data-quality.md).

## Results

Labels say what the evidence shows, not a verdict on the future. Full write-ups
and numbers are in [docs/findings/](docs/findings/README.md).

| label | strategies |
| --- | --- |
| **Supported** | USTEC risk-managed long (narrow claim: about half the drawdown of holding at the same Sharpe); intraday volume forecast (forecast holds, no execution benefit shown) |
| **Unproven** - could work, this data cannot decide | TOP8_2026 and the gold session breakout (a bet on current volatility); intraday momentum on USTEC (faded out of sample); 5- and 15-minute opening range (profit from the trade shape, not the direction); structural-break entry; overnight drift (real, too small); pre-FOMC drift (43 meetings) |
| **Failed a held-out test** | Tokyo gotobi fix; USDJPY carry; discovery program H01-H16 |
| **No edge found** | XAUUSD (10 families); EURUSD (12 families); round numbers; gold VWAP/EMA; New York open EMA; small-cap strategies; machine-learned FX; news breakout + LLM; BTC intraday |
| **Contradicted** - the effect points the wrong way | gold price levels; close rebalancing; overnight-intraday reversal |
| **Loses exactly its cost** - highest confidence | Precision Sniper; engulfing candle; sweeps and order blocks; pause bar; decision trees; engulfing quadrants (cannot have an edge as specified) |

Closest to deployment, by strength of evidence:

| rank | strategy | dev / validation / test | status |
| --- | --- | --- | --- |
| 1 | [USTEC risk-managed long](docs/findings/ustec-risk-managed-long.md) | Sharpe 0.95 / 0.91 / 1.15; max DD -15.1 / -12.9 / -7.9% | **deployable**, expert in `mt5/`; overnight swap unmeasured |
| 2 | [Intraday momentum, USTEC](docs/findings/intraday-momentum-and-btc-dynamics.md) | Sharpe 1.13 / 0.84 / 0.44 | forward-test candidate; decaying, no expert yet |
| 3 | [15-minute opening range, USTEC](docs/findings/orb15.md) | +0.223 / +0.202 / +0.086 R | forward-test candidate; direction does no work |
| 4 | [TOP8_2026, XAUUSD](docs/findings/session-breakout-top8.md) | R/trade -0.040 dev, +0.054 val, +0.147 in 2026 | unproven; loses 2020-2023, expert has flatten defects |
| 5 | [Pre-FOMC drift, USTEC](docs/findings/scheduled-flows.md) | +32.6 / +34.6 bps a meeting, t 1.42 | watch item; eight trades a year |

## Setup

Requires Python 3.11+ with polars, pyarrow, pandas, numpy, duckdb.

```bash
pip install -e .[dev]    # add ml (sklearn, xgboost, torch) or scripts (openpyxl, reportlab, yfinance) as needed
```

Raw vendor CSVs go in `Tick_Data/`. Everything under `data/` and `reports/` is
derived from them and can be rebuilt; neither is committed.

```bash
python scripts/pipeline/convert_ticks.py -w 8    # CSV -> parquet, resumable
python scripts/pipeline/build_bars.py -w 8       # 1m bars, all symbols
python scripts/pipeline/build_cost_model.py -w 8 # measure spread and latency
```

See [docs/commands.md](docs/commands.md) for the rest.

## Development workflow

| Stage | Purpose | Status |
| --- | --- | --- |
| 0 | Infrastructure: storage layout, CSV->Parquet pipeline, data quality report | **done** |
| 1 | Data layer: cleaning policy, bar construction, point-in-time loaders | **done** |
| 2 | Cost model: spread, commission, slippage | **done** |
| 3 | Research: hypothesis, feature/signal prototyping on the dev split | **done** (USTEC) |
| 4 | Backtest engine: signal / sizing / execution separation | **done** (tick-level) |
| 5 | Evaluation: tearsheets, cost drag, parameter sensitivity | **done** |
| 6 | Validation: held-out test, regime breakdown, stress tests | USTEC overlay **supported**, gold breakout **unproven** |
| 7 | Deployment readiness: paper trading, monitoring, kill switch | expert built, **not yet live** |

## Layout

```
Tick_Data/            raw vendor CSVs (~48 GB, gitignored, read-only)
data/                 derived parquet: ticks, bars, cost profiles, external (gitignored)
reports/              generated output: metrics, trade tapes, HTML (gitignored)
docs/                 the written record - prose, versioned
  findings/           one write-up per hypothesis tested
mt5/                  MetaTrader 5 experts for the deployable strategies
src/qlab/             the package
scripts/
  pipeline/           build the data: convert, bars, quality, cost model
  backtests/          one entry point per strategy
  research/           studies that are not a single strategy
tests/
```

`docs/` is written by hand and committed. `reports/` is written by scripts and
is not - if a number in `docs/` matters, the script that produced it is named
next to it.

## Package layout

```
src/qlab/
  paths.py           canonical filesystem layout - nothing hard-codes a location
  symbols.py         instrument specs and broker contract terms
  convert.py         CSV -> Parquet conversion
  quality.py         data quality metrics
  loader.py          cleaning policy, splits, point-in-time loading
  bars.py            bar construction and resampling
  session.py         session, rollover and reopen flags
  costprofile.py     measuring spread and latency drift
  costs.py           the cost model every backtest consumes
  engine.py          trade accounting, sizing, account rules
  metrics.py         tearsheets and cost decomposition
  stats.py           Newey-West errors, Lo's Sharpe standard error, the
                     stationary block bootstrap, and the multiple-testing
                     corrections every study is corrected with
  levels.py          round numbers and prior-period extremes
  eventstudy.py      forward-return distributions from an event table
  rollover.py        the price half of the overnight basis
  macro_calendar.py  scheduled US releases, reconstructed from publication rules
  microstructure.py  intraday activity and the periodic volume curve
  volume_spar.py     intraday volume by panel within-transformation
  propfirm.py        funded-account rules as a first-passage problem
  strategies/        one module per hypothesis
```
