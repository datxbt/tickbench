# tickbench

Tick-data research platform for systematic strategy development on Exness
raw-spread feeds (EURUSD, USDJPY, XAUUSD, USTEC), 2020-2026.

The installed distribution is `tickbench`; the import name is `qlab`.

Twenty hypotheses - price-action folklore, published papers, machine-learned
signals - implemented against 696M ticks and priced with a cost model measured
from the same feed. **One survived as a deployable strategy, one more as a
forecasting model.** The rejections are the product: they are written up in
full, with the null they were tested against.

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

## Setup

Requires Python 3.11+ with polars, pyarrow, pandas, numpy, duckdb.

```bash
pip install -e .
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
| 6 | Validation: held-out test, regime breakdown, stress tests | breakout **rejected**, USTEC overlay **accepted** |
| 7 | Deployment readiness: paper trading, monitoring, kill switch | expert built, **not yet live** |

## Layout

```
Tick_Data/            raw vendor CSVs (~48 GB, gitignored, read-only)
data/                 derived parquet: ticks, bars, cost profiles (gitignored)
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
