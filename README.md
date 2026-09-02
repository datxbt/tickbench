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
  symbols.py                   instrument specs (digits, pip size, asset class)
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
660M ticks, full rebuild in about a minute on 8 workers.

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

**All four symbols are usable.** The corpus agrees with the broker's published
contract specification wherever that specification is on file.

- **Zero spread on the majors is correct, not a defect.** EURUSD quotes
  `bid == ask` on 97.6% of ticks and USDJPY on 92.7%. This is what an Exness Raw
  Spread account *is*: the published average spread for both is 0.0 pips and the
  broker takes revenue as commission instead. Observed Mon-Fri means of 0.039 and
  0.071 pips agree with that spec.
- **Commission is the cost, not spread.** At $2.5 per lot per side, round-turn
  commission is 0.50 pips on EURUSD and about 0.76 on USDJPY (rate-dependent,
  since a JPY pip is worth a variable number of dollars). That is **93% and 91%
  of total round-turn cost** - a spread-only cost model would understate the
  true cost roughly tenfold.
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

### Open item

`symbols.py` carries the Exness Raw Spread FX terms. **XAUUSD and USTEC
commission and contract size are not filled in** - metals and indices sit on
separate specification pages. Until they are, cost estimates for those two cover
spread only, and `SymbolSpec.commission_pips()` raises rather than guessing.
