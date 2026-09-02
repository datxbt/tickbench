# QuantProject

Tick-data research platform for systematic strategy development on Exness
raw-spread feeds (EURUSD, USDJPY, XAUUSD, USTEC), 2020-2026.

## Development workflow

| Stage | Purpose | Status |
| --- | --- | --- |
| 0 | Infrastructure: storage layout, CSV->Parquet pipeline, data quality report | **done** |
| 1 | Data layer: cleaning policy, bar construction, point-in-time loaders | **done** |
| 2 | Cost model: spread, commission, slippage | next |
| 3 | Research: hypothesis, feature/signal prototyping on the dev split | |
| 4 | Backtest engine: signal / sizing / execution separation | |
| 5 | Evaluation: tearsheets, cost drag, parameter sensitivity | |
| 6 | Validation: held-out test, regime breakdown, stress tests | |
| 7 | Deployment readiness: paper trading, monitoring, kill switch | |

## Layout

```
Tick_Data/                     raw vendor CSVs (~48 GB, gitignored, read-only)
data/processed/
  ticks/symbol=<SYM>/                 converted parquet, one file per month
  bars/symbol=<SYM>/interval=<IVL>/   built bars, one file per month
  manifest.parquet                    inventory of every converted file
reports/data_quality/          Stage 0 quality report
src/qlab/                      the package
  paths.py                     canonical filesystem layout
  symbols.py                   instrument specs and broker contract terms
  convert.py                   CSV -> Parquet conversion
  quality.py                   data quality metrics
  loader.py                    cleaning policy, splits, point-in-time loading
  bars.py                      bar construction and resampling
  session.py                   session, rollover and reopen flags
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

# Build bars from the converted ticks (resumable; ~20 s for all four symbols)
python scripts/build_bars.py -w 8               # 1m bars, all symbols
python scripts/build_bars.py -i 1m 5m -s XAUUSD
```

Reading data downstream - always through `qlab.loader`, never by globbing
parquet directly, so that the cleaning policy and the split guard apply:

```python
from datetime import timedelta
from qlab.loader import load_bars, load_ticks
from qlab.bars import resample_bars
from qlab.session import with_session_flags
from qlab.symbols import get_spec

# Bars are small enough to load eagerly
bars = load_bars("XAUUSD", "1m", split="dev")

# Indicators warm on history before the split, and that history is flagged
bars = load_bars("XAUUSD", "1m", split="validation", warmup=timedelta(days=5))
evaluable = bars.filter(~bars["is_warmup"])

# Coarser bars come from the 1m bars, not from re-reading ticks
hourly = resample_bars(bars, "1h")

# Ticks are big: prefer lazy, and let the date filter prune whole months
ticks = load_ticks("EURUSD", start="2024-06-03", end="2024-06-07")
lazy = load_ticks("XAUUSD", lazy=True)

# Flag the expensive windows rather than dropping them
flagged = with_session_flags(bars, get_spec("XAUUSD"))
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

Bars add another 391 MB for 9.4M one-minute bars across the four symbols, built
in about 20 seconds:

| column | notes |
| --- | --- |
| `ts` | **the instant the bar's contents became known** - the interval *end* |
| `ts_open` | interval start |
| `first_tick_ts`, `last_tick_ts` | observed extent, so a 2-tick bar is visible as one |
| `open`, `high`, `low`, `close` | on the mid |
| `mid_mean` | tick-weighted mean mid |
| `bid_close`, `ask_close` | you enter at the ask and exit at the bid |
| `spread_mean`, `spread_close` | price units; divide by `spec.pip` for pips |
| `n_ticks` | quote updates in the interval |

There is no volume: the feed quotes bid and ask with no size, so volume and
dollar bars are not constructible. `bars.tick_bars()` is the substitute.

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

## The data layer

Stage 0 stored the vendor feed without judging it. Everything below is a
judgement, which is why each one is a named object you can inspect, override and
test rather than a line of code inside a loader.

### Point-in-time convention

**`ts` is the instant a row's contents became known.** A one-minute bar covering
`[10:00, 10:01)` is labelled `10:01`.

Right-edge labelling is chosen over the commoner left-edge convention because it
makes the naive thing safe: joining on `ts`, or acting on row *i* at row *i*'s
timestamp, uses only information that existed at that moment. Under left-edge
labels the identical code trades on a candle that has not closed yet - the most
common lookahead bug there is, and one that flatters a backtest rather than
breaking it. `ts_open` carries the left edge for anyone who needs it.

Two consequences worth knowing:

- **Empty intervals are missing rows, not flat bars.** Weekends, the daily
  maintenance break and outages simply have no bar. A flat bar is a price
  assertion, and there is no evidence for it; code that needs a regular grid
  should reindex explicitly so the filling is its own visible decision.
- **Session flags key off `ts_open`.** A bar covering `[20:59, 21:00)` is
  labelled `21:00`, so flagging it from the label would file it under the
  rollover hour when none of its ticks were in it.

### Duplicate timestamps

Every timestamp in the corpus is a whole millisecond - there is no sub-millisecond
precision anywhere in 696M ticks. When the market moves fast enough for two quote
updates to land inside one millisecond, the feed stamps them identically. They
are sequential real quotes, not repeats:

| | share of ticks that repeat the previous tick | share sharing a millisecond with a *different* quote |
| --- | ---: | ---: |
| EURUSD | 0.07% | 0.19% |
| USDJPY | 0.06% | 0.38% |
| XAUUSD | 0.10% | 0.43% |
| USTEC | 0.11% | 2.38% |

And it concentrates in fast markets: USTEC hit 9.0% in 2025-10 and gold 9.7% in
the March 2020 crash, against ~0.00% in quiet months. So collapsing to a unique
timestamp is a **volatility-correlated deletion** - it shaves highs and lows off
precisely the fastest bars. Measured on the worst months it changes 6% of bars,
moves mean bar range by about -0.15%, and takes up to 36 pips off a single gold
bar.

The default therefore keeps them and drops only true repeats. `UNIQUE_TS_POLICY`
is available for consumers that genuinely need a unique index, such as an as-of
join, and it states in its name what it is buying.

### Cleaning policy

`CleaningPolicy` is the whole of it, and `cleaning_report()` counts what any
policy would remove before you commit to it. Defaults: drop repeated rows
(lossless), keep distinct quotes at a shared millisecond, drop crossed
(`ask < bid`) and non-positive quotes. The corpus contains **zero** crossed
quotes, so those two rules are guards against a future feed rather than fixes
for this one.

Nothing about spread is cleaned. The Sunday reopen and the 21:00 rollover are
real prices at which real orders fill, and on the majors they hold most of the
non-zero spread in the sample - a strategy has to be able to see them to decide
to sit them out. `qlab.session` flags them instead.

### Splits

Fixed 2026-09-02, before any hypothesis existed, because a split chosen after
seeing results is not a split:

| split | range | purpose |
| --- | --- | --- |
| dev | 2020-01-29 to 2023-12-31 | hypothesis generation, features, parameter search |
| validation | 2024-01-01 to 2025-06-30 | model selection among survivors of dev |
| test | 2025-07-01 to 2026-09-01 | **locked** - one final estimate, run once |

Loading `test` raises `SplitLockedError` unless you pass `allow_test=True`. That
is not security, it is friction: it makes touching the held-out set a deliberate
act that shows up in a diff.

`warmup=` loads history *before* a split so indicators are warm at its first
bar, and flags those rows `is_warmup` so they can never be evaluated. Without it
the first N bars of every split are silently wrong; with it but no flag, the
leakage just moves somewhere harder to see.


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
- **Duplicate timestamps are mostly *not* duplicate rows.** ~~The audit read
  0.25-2.5% of shared timestamps as exact duplicate rows, and concluded that
  deduplication would be lossless.~~ Corrected in Stage 1: the feed stamps to
  the millisecond and nothing finer, so a fast market puts several genuine
  quotes on one timestamp. Only 0.09% of the corpus is a true repeat of the
  preceding tick; 1.0% shares a millisecond with a *different* quote. See
  "Duplicate timestamps" below.
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
