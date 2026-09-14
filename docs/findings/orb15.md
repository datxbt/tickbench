# The 15-minute opening range, pre-registered - evaluation

Subject: the one follow-up [opening-range-breakout.md](opening-range-breakout.md)
named as legitimate. The 15-minute range beat the paper's 5 minutes on both
splits there, and the report said: fix it, and test it once on `test`. This
study does exactly that - `ORBConfig(or_minutes=15)`, every other field the
paper's (the candle's sign picks the side, a stop order at the range edge, a
protective stop 10% of the 14-day ATR away, hold to 16:00 New York), USTEC,
resolved on ticks by `qlab.strategies.opening_range`.

Driver: `scripts/backtests/backtest_orb15.py` for dev and validation with
controls, and `--test` for the single locked-split run.

**Verdict: not deployable. On test the rule made +0.086 R a trade at t = +0.57 -
positive, a third of its dev and validation estimate, and inconclusive under the
rule written before the split was read. The test split is now spent for the
opening-range family.**

---

## 1. Dev and validation, with the controls that matter

| | n | mean R | t (HAC) | hit | shuffled-sign placebo | both sides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 854 | +0.223 | +2.20 | 19.8% | +0.178 (p 0.26) | +0.171 |
| validation | 331 | +0.202 | +1.14 | 19.9% | +0.078 (p 0.23) | +0.231 |
| **pooled** | **1,185** | **+0.217** | **+2.46** | 19.8% | | |

The 5-minute study's finding repeats at 15 minutes: **the direction does no
measurable work.** Shuffling the candle's sign while keeping the calendar, the
ranges, the ATR and the stop earns most of the edge, and taking both sides earns
as much as the rule on validation. What is left is geometry - a tight stop and an
uncapped hold to the bell, paid for by the one session in five that trends. That
is still a return, so the cell went to test on its pooled t of +2.46 with a
stated low-power warning: at +0.2 R and a per-trade sd near 2.3 R, ~290 test
sessions give an expected t near 1.5.

## 2. The test split, run once

The rule, fixed in the script's docstring before the run: PASS at a net mean > 0
with one-sided t > 1.28; FAIL at a mean <= 0; inconclusive between.

| | n | mean R | t (HAC) | hit | mid R |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 256 | **+0.086** | +0.57 | 21.1% | +0.126 |
| H2 2025 | 108 | +0.262 | | | |
| 2026 | 148 | -0.042 | | | |

Inconclusive, and read with the rest of the series it is the decay the 5-minute
report described: +0.28 R in 2021 and 2023, +0.09 in 2024, +0.06 in H1 2025,
+0.26 in H2 2025 and slightly negative in 2026. A geometry edge that pays for
itself on the one trending session in five is exposed to exactly how often
sessions trend, and that is not something the rule controls.

```bash
python scripts/backtests/backtest_orb15.py          # dev + validation, with controls
python scripts/backtests/backtest_orb15.py --test   # already run; the split is spent
```
