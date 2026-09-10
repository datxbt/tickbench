# Data quality


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

## Contract terms on file

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
