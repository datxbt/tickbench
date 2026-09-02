# QuantProject

Tick-data research platform for systematic strategy development on Exness
raw-spread feeds (EURUSD, USDJPY, XAUUSD, USTEC), 2020-2026.

## Development workflow

| Stage | Purpose | Status |
| --- | --- | --- |
| 0 | Infrastructure: storage layout, CSV->Parquet pipeline, data quality report | **done** |
| 1 | Data layer: cleaning policy, bar construction, point-in-time loaders | next |
| 2 | Cost model: spread, commission, slippage | |
| 3 | Research: hypothesis, feature/signal prototyping on the dev split | |
| 4 | Backtest engine: signal / sizing / execution separation | |
| 5 | Evaluation: tearsheets, cost drag, parameter sensitivity | |
| 6 | Validation: held-out test, regime breakdown, stress tests | |
| 7 | Deployment readiness: paper trading, monitoring, kill switch | |

## Layout

```
Tick_Data/                     raw vendor CSVs (~48 GB, gitignored, read-only)
data/processed/
  ticks/symbol=<SYM>/          converted parquet, one file per month
  manifest.parquet             inventory of every converted file
reports/data_quality/          Stage 0 quality report
src/qlab/                      the package
  paths.py                     canonical filesystem layout
  symbols.py                   instrument specs and broker contract terms
  convert.py                   CSV -> Parquet conversion
  quality.py                   data quality metrics
scripts/                       command-line entry points
```

## Setup

Requires Python 3.11+ with polars, pyarrow, pandas, numpy, duckdb.

```bash
pip install -e .
```

## Usage

```bash
# Convert raw CSVs to parquet (resumable; skips files already up to date)
python scripts/convert_ticks.py                  # all symbols
python scripts/convert_ticks.py -s XAUUSD --force
python scripts/convert_ticks.py -w 8             # worker processes

# Assess data quality and regenerate reports/data_quality/DATA_QUALITY.md
python scripts/quality_report.py -w 8
```

Reading ticks downstream:

```python
import polars as pl
from qlab import paths

# one month
df = pl.read_parquet(paths.tick_parquet_path("XAUUSD", 2024, 6))

# a whole symbol, lazily
lf = pl.scan_parquet(paths.tick_partition_dir("XAUUSD") / "*.parquet")

# every symbol, with the partition key recovered as a column
lf = pl.scan_parquet(paths.TICKS_DIR / "**/*.parquet", hive_partitioning=True)
```

## Storage format

Each monthly parquet holds three columns:

| column | type | notes |
| --- | --- | --- |
| `ts` | `Datetime("us", "UTC")` | ascending, timezone-aware |
| `bid` | `Float64` | |
| `ask` | `Float64` | |

The vendor's constant `Exness` and `Symbol` columns are verified then dropped;
the symbol lives in the hive partition path. `mid` and `spread` are deliberately
not stored - they are one cheap expression away and would inflate every file.

zstd level 3 compresses the corpus roughly 10x: **48 GB CSV -> 4.6 GB parquet**,
696M ticks, full rebuild in about a minute on 8 workers.

## Design rules

**Conversion is lossless.** `convert.py` verifies structural invariants (constant
symbol column, no null timestamps, ascending order) and fails loudly on
violation, but it never deduplicates, filters or repairs. Every such decision is
a modelling choice and belongs in Stage 1 where it is explicit and reversible -
if conversion silently fixed the data, every backtest would inherit an invisible
assumption.

**Quality measurement is separate from conversion.** QC reads parquet, so checks
can be added and re-run in minutes without touching the 48 GB source.

**Writes are atomic.** Parquet is written to a temp file and renamed, so an
interrupted run can never leave a truncated file that the resume logic would
mistake for finished work.

## Data quality findings

See `reports/data_quality/DATA_QUALITY.md` for the full report.

**All four symbols are usable and agree with the broker's published contract
specification.**

- **Zero spread on the majors is correct, not a defect.** EURUSD quotes
  `bid == ask` on 97.6% of ticks and USDJPY on 92.7%. This is what an Exness Raw
  Spread account *is*: the published average spread for both is 0.0 pips and the
  broker takes revenue as commission instead. Observed trailing-3-month Mon-Fri
  means of 0.029 and 0.042 pips agree with that spec.
- **Commission is the cost on FX, but not everywhere.** Round-turn commission is
  0.50 pips on EURUSD and ~0.80 on USDJPY (rate-dependent, since a JPY pip is
  worth a variable number of dollars) - **95% of total cost on both**. On gold
  it is 44% and on USTEC 60%, because those carry a real spread. One blanket
  cost assumption across the four would be wrong in both directions.
- **In basis points, all four cost about the same.** Pips are not comparable
  across instruments; bps of price are. Round turn: EURUSD 0.46, USDJPY 0.53,
  USTEC 0.36, XAUUSD 0.37 bps. Instrument choice should therefore turn on edge
  and volatility, not on cost.
- **Spread is regime-dependent, and the regimes are sharp.** Weekdays run ~2-5%
  non-zero spread, but the Sunday reopen hits 37% (EURUSD) and 53% (USDJPY), and
  21:00 UTC - the daily rollover - averages 1.5 and 3.3 pips. Avoiding those two
  windows matters far more than anything else in the FX cost model.
- **Spreads have compressed since 2020.** XAUUSD's median went 11.3 -> 3.7 pips
  and USDJPY's zero-spread share went 73% -> 96%. Cost is time-varying; a single
  spread constant applied across 2020-2026 will be wrong at both ends.
- **Gaps are mostly the daily maintenance break, not outages.** XAUUSD and USTEC
  each show ~1,600 routine session breaks; once those and weekends are excluded,
  genuine hour-plus outages fall to 51 and 65 over six years. FX has no daily
  break, so its 16 and 12 hour-plus gaps are all real downtime.
- **Duplicate timestamps are benign.** 0.25-2.5% of ticks share a timestamp with
  the previous one, but they are exact duplicate rows, so deduplication in
  Stage 1 will be lossless.
- **The spec check must use a trailing window.** Exness publishes a
  previous-trading-day average, so comparing it against a multi-year mean
  produces false alarms on anything whose spread has moved. Gold averages 8.74
  pips over the trailing three months against a published 9.0 - agreement - but
  6.0 over 2024-2026, because its spread ran 5.8 -> 3.7 -> 9.0 across those
  years. `spec_agreement()` therefore takes `trailing_months`, not a start year.

### Contract terms on file

All four symbols now carry their published Exness Raw Spread terms:

| Symbol | Published avg spread | Commission/side | Contract | Commission round turn |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | 0.0 pips | $2.50 | 100,000 | 0.500 pips |
| USDJPY | 0.0 pips | $2.50 | 100,000 | ~0.80 pips (rate-dependent) |
| XAUUSD | 9.0 pips | $3.50 | 100 oz | 7.000 pips |
| USTEC | 0.6 pips | $0.313 | 1 index point | 0.626 pips |

Spreads and commissions are published figures. **Contract sizes are not** - gold
uses the standard 100 oz lot, and USTEC's single-unit contract is inferred from
Exness listing `USTEC_x100` at exactly 100x the commission ($31.3 vs $0.313).
Both are worth confirming against the account's own contract specification,
since every pip-denominated cost scales with them.
