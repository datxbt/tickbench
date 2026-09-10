# The data layer

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

## Loading conventions

Stage 0 stored the vendor feed without judging it. Everything below is a
judgement, which is why each one is a named object you can inspect, override and
test rather than a line of code inside a loader.

## Point-in-time convention

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

## Duplicate timestamps

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

## Cleaning policy

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

## Splits

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

