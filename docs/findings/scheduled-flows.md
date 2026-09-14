# Scheduled flows: macro releases and the FX fixings - evaluation

Subject: a search for an edge in mechanisms this project had not yet tested,
run on all four instruments. Every earlier study converged on the same two
constraints - minute-level predictability is about 20x smaller than the round
turn, and a daily strategy cannot be validated on ~1,000 sessions - so the
search looked where both could be escaped: **scheduled events with a named
counterparty**, where the effect per event can be large and the count of events
can be high.

Eight hypotheses in three pre-registered waves. Each wave's primary cells, their
predicted sign and their control were written into the script's docstring before
it was run.

**Verdict: rejected. One candidate - the unwind after the Tokyo fix on gotobi
days, USDJPY - passed everything dev and validation could throw at it and then
failed the locked test split: +2.85 bps a trade at t = +4.38 over 2020 to mid
2025, -0.51 bps at t = -0.44 over the fourteen months that followed. The test
split is spent for this family. Everything else failed earlier.**

---

## 0. Method

Every trade is a clock rule on a calendar known in advance: enter at the
one-minute bar-close quote standing at a stated instant, exit at another, long
pays the ask and leaves at the bid. Commission and two sides of measured
slippage come from the split's own cost model. Every result is reported at the
mid and net, pooled over dev and validation and per split. The control for each
hypothesis is the **same rule on the same clock minutes on days with no event**
- the mechanics-only null that separates "the event does it" from "the clock
does it".

The macro calendar is `qlab.macro_calendar`, whose reconstruction was audited
against the tape in [news-breakout-and-volume-spar.md](news-breakout-and-volume-spar.md).
The FOMC dates are the Fed's published schedule; the two March 2020
inter-meeting cuts are excluded from every pre-announcement test, since nobody
could have positioned for them.

## 1. Wave 1: four macro-event hypotheses

`scripts/research/macro_event_screen.py`, 35 cells, Bonferroni |t| >= 3.19.

| hypothesis | rule | predicted | result |
| --- | --- | --- | --- |
| H1 pre-FOMC drift (Lucca & Moench 2015) | long 14:00 ET day before -> 13:55 ET statement day | USTEC up | USTEC **+33.1 bps**, t +1.42; dev +32.6, val +34.6 |
| H2 FOMC statement fade | fade 14:00 -> 14:25 ET, hold to 15:55 | reversal | positive on all four, t +1.0 to +1.4 |
| H3 post-release drift, CPI and NFP | follow 08:30 -> 08:35 ET, hold 60/120/240 m | continuation | USTEC **reverses** in 6 of 6 cells; NFP 120 m t -2.52 |
| H4 month-end hedge rebalancing (Melvin & Prins 2015) | 12:00 -> 16:00 London, sell USD if USTEC up MTD | USD down | -5.5 / -4.5 / +6.1 bps, nothing past t 1.3 |

Nothing clears the correction. Three patterns were consistent enough to test on
fresh events: H3's reversal on USTEC, H2's fade, and H1 - which lands on the
published magnitude in both splits but has only 43 meetings and no second sample
to confirm on. H1 is a regularity this corpus cannot establish, not one it
refutes.

## 2. Wave 2: confirmation on events wave 1 never touched

`scripts/research/macro_event_confirm.py`.

**The USTEC 08:30 fade fails, with the wrong sign.** Confirmation set: the 349
dev/validation days carrying an 08:30 jobless-claims, PPI, retail-sales or GDP
release and no CPI or NFP. Pre-registered primary cell (fade 08:30 -> 08:35,
enter 08:35, exit 10:35):

| | n | net bps | t |
| --- | ---: | ---: | ---: |
| confirmation set | 347 | **-7.09** | -1.71 |
| CPI/NFP (in-sample, wave 1) | 125 | +20.88 | +2.89 |

The dose-response the mechanism needs is absent on the in-sample days too:
regressing the 08:35 -> 10:35 return on the 08:30 -> 08:35 move gives beta -0.093
at t -0.46 on CPI/NFP days. And the in-sample exit curve is jagged rather than
monotone - +2.9 bps to 09:30, +12.4 to 09:45, +8.4 to 10:00, +20.9 to 10:35 -
which is noise, not a reversal completing. Wave 1's reversal was a
125-event artefact.

**The FOMC fade does not transfer to the minutes.** Minutes print at 14:00 ET
three weeks after each scheduled meeting - the same clock minute, no press
conference. Fading them gives +1.7 / +0.4 / +1.5 / **-3.7** bps across EURUSD,
USDJPY, XAUUSD and USTEC, none past t 1.35. If the statement-day fade is real it
belongs to the press conference, and 44 meetings cannot say whether it is.

## 3. Wave 3: order flow at the FX fixings

`scripts/research/fx_fix_screen.py`, 20 cells, Bonferroni |t| >= 3.02. Two
benchmark prices at which non-discretionary customer flow executes:

- **The Tokyo fix on gotobi days** (Ito & Yamada 2017). Japanese importers settle
  on the 5th, 10th, 15th, 20th, 25th and month-end and buy USD at the 09:55 JST
  fix. G1: long USDJPY 09:00 -> 09:55 JST. G2: short 09:55 -> 11:00 JST, the
  unwind. Control: non-gotobi weekdays.
- **The WM/R 16:00 London fix** (Evans 2018). F1: fade 15:30 -> 16:00, enter
  16:03, exit 17:00, every weekday. F2: F1 at month-end only. Control: the same
  fade one clock hour earlier.

| cell | n | net bps | t | dev t | val t | vs control t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| USDJPY G1 into the fix | 382 | -0.32 | -0.44 | +0.38 | -1.42 | -0.35 |
| **USDJPY G2 the unwind** | **388** | **+2.85** | **+4.38** | **+3.15** | **+3.17** | **+3.14** |
| EURUSD G2 | 388 | -0.48 | -1.17 | +0.80 | -4.23 | -0.84 |
| XAUUSD G2 | 383 | +0.42 | +0.36 | -0.02 | +0.60 | +0.16 |
| USTEC G2 | 382 | -1.32 | -1.24 | -1.52 | +0.28 | +0.38 |
| USDJPY F1 WM/R fade | 1,400 | -0.98 | -2.88 | -2.17 | -1.97 | +0.14 |
| EURUSD F1 WM/R fade | 1,402 | -0.39 | -1.28 | -0.56 | -1.65 | -0.65 |

The WM/R fade loses on every instrument. G2 is the only cell in the three waves
to clear its multiplicity correction, it is significant in each split on its
own, it beats its same-clock control, and it is **specific to the instrument the
mechanism names** - the three falsification instruments show nothing. Curiously
the run-up into the fix (G1) is not there; only the unwind is.

## 4. The robustness battery, on dev and validation

`scripts/research/gotobi_robustness.py`. Nothing here changes G2's
specification; it asks only whether G2 is fragile.

**Entry x exit surface**, pooled net bps (t):

| entry \ exit | 10:15 | 10:30 | 11:00 | 11:30 | 12:00 | 13:00 | 15:00 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 09:55 | +2.34 (+5.39) | +2.73 (+5.13) | **+2.85 (+4.38)** | +3.04 (+4.00) | +2.34 (+2.93) | +3.81 (+3.11) | +4.29 (+2.83) |
| 09:56 | +1.29 (+3.27) | +1.69 (+3.40) | +1.79 (+2.87) | +1.99 (+2.70) | +1.29 (+1.65) | +2.75 (+2.26) | +3.24 (+2.14) |
| 10:00 | +0.92 (+2.58) | +1.32 (+2.86) | +1.43 (+2.42) | +1.63 (+2.32) | +0.92 (+1.24) | +2.39 (+1.99) | +2.88 (+1.91) |
| 10:05 | +0.04 (+0.13) | +0.44 (+1.05) | +0.56 (+0.99) | +0.79 (+1.15) | +0.06 (+0.09) | +1.59 (+1.34) | +2.11 (+1.42) |

Positive at every exit, and front-loaded: a one-minute delay costs ~1.06 bps
everywhere, and by 10:05 it is gone. Section 5 checks whether a real order can
get in.

- **Every year positive**: +3.65, +1.52, +3.55, +0.75, +4.19, +4.09 bps for 2020
  through H1 2025.
- **Weekday-matched permutation null**, 20,000 draws: real +2.845 against a null
  of -0.040 +/- 0.704, **p = 0.0001**. This matters because weekend-shifted
  gotobi dates land on Fridays - and non-gotobi Fridays return -0.79, so the
  Friday skew is not what produced it.
- **Adjacent-day placebos**, one and two business days either side, gotobi days
  excluded: +0.71, +0.87, -0.31, +0.83 bps, none past t 1.05.
- **Cost stress**: doubled spread, fully adverse slippage and 1000 ms latency at
  once leave **+2.45 bps, t +3.78**. USDJPY's Tokyo-morning cost is 0.64 bps and
  the spread paid is 0.10, so cost was never the question.
- **By day type** the big settlement dates carry it - 20th +7.41 (t 3.33), 25th
  +5.56 (t 3.45), month-end +5.17 (t 2.83) against the 5th +0.87 and 10th -1.07.
  That is what an importer-flow story predicts, and it was not used to refine
  the rule.
- **One mechanism check points the wrong way.** On the 24 gotobi days that fell
  on a Japanese public holiday - Tokyo shut, no fix - the unwind was +6.70 bps
  (t 2.03). The mechanism predicts zero there. It is a small, noisy subset
  (1.2 standard errors from the open-day +2.59), but it is the one observation
  in the battery that the story does not explain.

## 5. On ticks: the drop can be reached

`scripts/research/gotobi_ticks.py`, 5.9M USDJPY ticks in the 09:50-11:05 JST
window. Mean short-side mid path from 09:55:00, bps:

| seconds after the fix | 1 | 2 | 5 | 10 | 30 | 60 | 300 | 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gotobi | -0.02 | +0.37 | +0.56 | +0.68 | +0.85 | +1.05 | +1.42 | +2.24 |
| control | -0.01 | +0.18 | +0.29 | +0.26 | +0.20 | +0.26 | +0.44 | +0.69 |

The move builds over seconds and minutes rather than landing on the first tick.
G2 repriced at tick-accurate fills - sell the bid of the first tick at
09:55:00 + L, buy the ask at 11:00, commission charged:

| latency L | 0 s | 1 s | 5 s | 10 s | 60 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| net bps | +3.02 | **+2.94** | +2.42 | +2.30 | +1.96 |
| t | +4.61 | +4.49 | +3.71 | +3.55 | +3.11 |

At this point G2 was the strongest candidate this project has produced: a
pre-registered cell that cleared its correction, a named counterparty, the
right instrument and only that instrument, stable in every year, robust to
exit, cost and a realistic retail latency. That is what the test split is for.

## 6. The test split, run once

`scripts/research/gotobi_test.py`. The pass rule was written into the script
before the split was read: net mean > 0 with one-sided t > 1.28 (10%, because
~84 events at the dev+validation effect give an expected t near 2.0), fail at a
mean <= 0, inconclusive between.

| | n | net bps | sd | t | hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| bar rule, as registered | 83 | **-0.51** | 10.36 | -0.44 | 0.43 |
| ticks, 1 s latency | 83 | -0.56 | 10.30 | -0.50 | 0.45 |

**FAIL.** Not an unlucky draw from the same distribution: the test estimate sits
about 2.6 standard errors below dev+validation's.

Described after the verdict, and not used for anything: the whole Tokyo-fix
unwind went, not only the gotobi premium. On non-gotobi days the same short
returned +1.04 bps at the mid over dev+validation and **-0.38** over test; by
quarter the gotobi cell runs -0.42, +2.80, -3.43, -0.01, -1.67 bps. Something
changed at the 09:55 fix around mid-2025 - this corpus holds prices, not the
reason.

## 7. What carries forward

1. **The strongest candidate here died on the split that was never looked at.**
   Pre-registration, a Bonferroni pass, a permutation p of 0.0001, six positive
   years, a same-clock control, placebos, cost stress and tick fills are all
   evidence about 2020 to mid-2025. None of them is evidence about the next year.
   That is the argument for the locked split, made by the only candidate that
   got far enough to use it.
2. **Pre-FOMC drift on USTEC is the one untested-and-unrefuted regularity left.**
   +33 bps a meeting, the same in both splits, at the published magnitude, on 43
   events. It needs a longer sample than this corpus holds.
3. **Fading or following the first reaction to a scheduled release does
   nothing** on any of the four instruments once a fresh set of releases is used.

## 8. Reproduce

```bash
python scripts/research/macro_event_screen.py      # wave 1: FOMC, CPI/NFP, month-end
python scripts/research/macro_event_confirm.py     # wave 2: disjoint confirmation sets
python scripts/research/fx_fix_screen.py           # wave 3: Tokyo gotobi fix, WM/R fix
python scripts/research/gotobi_robustness.py       # the battery on dev + validation
python scripts/research/gotobi_ticks.py            # tick path and latency fills
python scripts/research/gotobi_test.py             # the locked split - already spent
```

Output lands in `reports/macro_event_screen/`.

## 9. Addendum: two more benchmarks, wave 4

`scripts/research/fix_benchmarks.py`, run after the gotobi test. The same fade
shape as the WM/R rule - sign of the 30 minutes into the benchmark, faded from
three minutes after it, out an hour later, control one clock hour earlier - at
two benchmarks with their own counterparties: the **Shanghai Gold Benchmark**
auctions (10:15 and 14:15 Beijing) and the **ECB reference rate** (14:15 CET).
Primary cells pre-registered at one-sided t > 2.13 (Bonferroni over three):

| cell | n | net bps | t | dev | validation | mid | control mid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAUUSD SGE-AM | 1,398 | -1.07 | -2.42 | -1.62 | +0.39 | -0.13 | +0.40 |
| XAUUSD SGE-PM | 1,396 | -0.76 | -1.64 | -1.15 | +0.26 | +0.21 | -0.19 |
| EURUSD ECB | 1,401 | -0.55 | -1.12 | -0.17 | -1.55 | +0.03 | +0.23 |

All three fail, and so does every falsification cell: across twelve
(instrument x benchmark) cells the mid never leaves +/-1 bp and never separates
from its control hour. The negative nets are the round turn. Neither auction
leaves a footprint a fade can collect.
