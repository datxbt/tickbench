# Stage 0 - Tick Data Quality Report

Generated from 321 symbol-months across 4 instruments.

This report measures the raw vendor feed. Nothing here has been cleaned or
repaired - the point is to decide what is trustworthy before any strategy
work begins.

## 1. Inventory

| Symbol | Class | Months | Rows | First tick | Last tick | Median ticks/day |
| --- | --- | ---: | ---: | --- | --- | ---: |
| EURUSD | fx | 80 | 86,878,906 | 2020-01-29 | 2026-08-31 | 41,722 |
| USDJPY | fx | 80 | 109,419,415 | 2020-01-29 | 2026-08-31 | 50,541 |
| USTEC | index | 81 | 216,564,864 | 2020-01-29 | 2026-09-01 | 57,182 |
| XAUUSD | metal | 80 | 283,059,760 | 2020-01-29 | 2026-08-31 | 103,512 |

## 2. Quote integrity

`locked` = ticks where bid == ask (zero spread). A raw-spread feed should
almost never be locked; a high figure means the ask side is not a real quote.
`crossed` = ask < bid, which is always corrupt.

| Symbol | Locked % | Crossed % | Dup ts % | Out-of-order files | Spread p50 (pips) | Spread p95 | Spread max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 97.58 | 0.0000 | 0.255 | 0 | 0.00 | 0.00 | 45.00 |
| USDJPY | 92.66 | 0.0000 | 0.443 | 0 | 0.00 | 0.05 | 35.00 |
| USTEC | 15.17 | 0.0000 | 2.493 | 0 | 0.59 | 1.09 | 28.60 |
| XAUUSD | 0.00 | 0.0000 | 0.529 | 0 | 6.30 | 6.30 | 396.30 |

## 3. Continuity

Weekend gaps and the instrument's daily maintenance break are excluded, so
the outage columns count only unexplained downtime. `Daily breaks` is the
routine broker session break, shown for reference - roughly one per trading
day is normal for XAUUSD and USTEC.

| Symbol | Missing weekdays | Daily breaks | Outages >1min | >5min | >1h | Longest outage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 2 | 0 | 57,634 | 766 | 16 | 34.6 h |
| USDJPY | 2 | 0 | 45,030 | 643 | 12 | 34.6 h |
| USTEC | 8 | 1,677 | 1,002 | 150 | 65 | 34.6 h |
| XAUUSD | 12 | 1,617 | 542 | 81 | 51 | 34.6 h |

## 4. Price sanity

A jump is a tick-to-tick mid move greater than 10 bps.

| Symbol | Price range | Jumps | Largest jump (bps) |
| --- | --- | ---: | ---: |
| EURUSD | 0.954 - 1.235 | 139 | 116.7 |
| USDJPY | 101.185 - 163.990 | 293 | 126.3 |
| USTEC | 6,629.760 - 30,795.110 | 2,956 | 597.0 |
| XAUUSD | 1,451.365 - 5,595.423 | 2,433 | 242.4 |

## 5. Verdict

- **EURUSD**: **ask side unusable** - 97.6% of ticks are locked (bid == ask), so the empirical spread distribution is not a real spread. Cost modelling for this symbol needs an external spread source; the bid series is still usable as a price series.; 2 weekdays with no data; 16 unexplained outages over an hour
- **USDJPY**: **ask side unusable** - 92.7% of ticks are locked (bid == ask), so the empirical spread distribution is not a real spread. Cost modelling for this symbol needs an external spread source; the bid series is still usable as a price series.; 2 weekdays with no data; 12 unexplained outages over an hour
- **USTEC**: locked ticks concentrated in 2024 (8%), 2025 (39%) - treat those years' spreads with care; 8 weekdays with no data; 65 unexplained outages over an hour
- **XAUUSD**: 12 weekdays with no data; 51 unexplained outages over an hour

## 6. UTC hour coverage

Share of each symbol's ticks by UTC hour - the basis for session definitions
in Stage 2 cost modelling.

| Symbol | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 2.8 | 3.2 | 2.4 | 2.0 | 1.7 | 2.2 | 4.0 | 6.0 | 6.1 | 5.1 | 4.5 | 4.7 | 7.1 | 8.7 | 9.3 | 7.7 | 5.2 | 4.0 | 4.1 | 3.5 | 2.1 | 1.0 | 1.1 | 1.4 |
| USDJPY | 5.5 | 4.9 | 3.6 | 3.1 | 2.9 | 3.2 | 4.4 | 5.6 | 5.4 | 4.3 | 3.7 | 3.8 | 6.2 | 7.8 | 8.1 | 6.5 | 4.3 | 3.5 | 3.5 | 3.0 | 1.8 | 1.0 | 1.4 | 2.5 |
| USTEC | 2.9 | 2.9 | 2.3 | 1.9 | 1.6 | 2.1 | 2.3 | 3.0 | 3.4 | 2.8 | 2.6 | 2.9 | 3.6 | 9.3 | 12.1 | 9.9 | 7.4 | 6.6 | 6.4 | 7.0 | 3.4 | 0.5 | 1.2 | 1.8 |
| XAUUSD | 3.5 | 5.4 | 4.0 | 3.1 | 2.5 | 3.5 | 4.2 | 4.2 | 4.2 | 3.8 | 3.5 | 3.9 | 5.8 | 8.5 | 8.7 | 7.0 | 5.2 | 4.4 | 4.1 | 3.8 | 2.6 | 0.7 | 1.4 | 2.3 |
