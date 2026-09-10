# USDJPY: the overlay that worked on USTEC, and did not work here

**Verdict: rejected on the held-out split.** The strategy that survived dev and
validation returned **−0.11% on test while simply holding returned +8.59%**, at
the same drawdown. The test split is now spent for this strategy family.

The arc is the whole story:

| | dev 2020-23 | validation 2024-25H1 | test 2025H2-26 |
| --- | ---: | ---: | ---: |
| **strategy Sharpe** | **0.81** | **0.26** | **0.01** |
| strategy CAGR | 4.93% | 1.40% | −0.11% |
| strategy max drawdown | −10.8% | −8.0% | −4.2% |
| buy & hold Sharpe | 0.67 | 0.09 | **1.08** |
| buy & hold CAGR | 6.16% | 0.41% | **8.59%** |
| buy & hold max drawdown | −15.6% | −13.9% | −4.5% |

Zero swap; costs from the measured tape. A monotone decay from 0.81 to 0.26 to
0.01 is what an effect that was never there looks like when it meets three
successive periods of data.

- Research driver: `scripts/backtests/backtest_carry_harvest.py`
- Model: `qlab.strategies.risk_managed_long` with `session=FX_DAY`
- Raw numbers: `reports/strategies/carry_harvest.json`

---

## 1. Why USDJPY looked like the best of the remaining three

Unlike EURUSD, USDJPY **has a drift** in this corpus: the yen fell from 108 to
above 150, and buy-and-hold returned 6.16%/yr over dev. That is the ingredient
the EURUSD study could not find, and it is what made the USTEC recipe worth
trying — there was something to size.

And unlike USTEC, **the swap runs in the trader's favour.** A long USDJPY
receives carry. `qlab.rollover` measures the price side of it at −1.47 bps a
night, so a long carries about −3.8%/yr of price drag that a swap credit offsets.
Every number in this report is run at zero swap, which on USTEC *flatters* a
result and here **understates** it, by up to 3.8 percentage points a year.

So: a real drift, a favourable financing asymmetry, and machinery that already
worked next door. It still failed.

---

## 2. The diagnosis: the sizing rule levers instead of shrinking

Pointing the USTEC configuration straight at USDJPY makes the drawdown **worse
than doing nothing**, in both splits:

| | buy & hold | USTEC config as-is |
| --- | ---: | ---: |
| dev max drawdown | −15.6% | **−19.2%** |
| validation max drawdown | −13.9% | **−15.3%** |
| dev average weight | 1.00 | 1.08 (1.71 in 2021) |

The reason is arithmetic. USTEC runs at 27% annualised volatility against a 15%
target, so the rule **shrinks** the position — that is where the drawdown
reduction comes from. USDJPY runs at 9.6%, so the same rule **levers** it to
1.5-1.7x. A risk manager pointed at a quiet instrument becomes a leverage
machine.

Hence the candidate set below caps the weight at 1.0 and drops the target to 8%,
close to USDJPY's own volatility. That is a structural correction, not a fit.

---

## 3. The candidate set

Five candidates, fixed before evaluation, each a different answer to "what
protects a carry position". The one risk USDJPY actually has is the carry
unwind, and volatility is a poor warning of it — carry crashes happen *because*
the calm attracted the leverage.

The candidate with the best story was **J4**: gate the yen position on equity
trend. The yen carry trade and long equity are the same risk-appetite bet held
two ways, they unwound together in March 2020 and August 2024, and this corpus
quotes them on the same clock. It is exactly the kind of cross-asset idea this
dataset exists to test.

**dev**

| candidate | CAGR% | vol% | SR | maxDD% | Calmar | avg w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J0 buy and hold | 6.16 | 9.59 | 0.67 | −15.6 | 0.39 | 1.00 |
| J1 yen trend gate | 4.86 | 7.48 | 0.67 | −15.1 | 0.32 | 0.62 |
| **J2 J1 + vol target** | 4.93 | 6.15 | **0.81** | **−10.8** | 0.46 | 0.56 |
| J3 J2 + equity risk gate | 1.36 | 4.15 | 0.35 | −6.7 | 0.20 | 0.38 |
| J4 equity risk gate only | 4.35 | 5.70 | 0.78 | −7.3 | **0.60** | 0.54 |
| J6 vol target, no gate | 5.17 | 7.56 | 0.71 | −10.9 | 0.47 | 0.86 |

**validation** — the split whose stated purpose is selection

| candidate | CAGR% | vol% | SR | maxDD% | Calmar | avg w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J0 buy and hold | 0.41 | 10.88 | 0.09 | −13.9 | 0.03 | 1.00 |
| J1 yen trend gate | 1.46 | 6.91 | 0.25 | −8.8 | 0.17 | 0.56 |
| **J2 J1 + vol target** | 1.40 | 5.96 | **0.26** | **−8.0** | 0.17 | 0.48 |
| J3 J2 + equity risk gate | 1.40 | 5.96 | 0.26 | −8.0 | 0.17 | 0.48 |
| J4 equity risk gate only | 0.39 | 10.04 | 0.09 | −13.9 | 0.03 | 0.88 |
| J6 vol target, no gate | 0.05 | 8.50 | 0.05 | −12.0 | 0.00 | 0.79 |

**J4 has the best Calmar in dev and is indistinguishable from buy-and-hold in
validation.** The equity gate does not fire in time: in August 2024 the Nasdaq
fell sharply but did not break its own 200-session average until after the yen
had already unwound. Measured over that window, J4 lost **−13.44%** against
buy-and-hold's −13.44% — it provided no protection at all, on the single
occasion it was designed for.

J2 was selected: best drawdown in validation, second-best Sharpe in dev, and the
only candidate that improves on buy-and-hold along both axes in both splits.

### It did protect against the unwinds

| | buy & hold | J2 |
| --- | ---: | ---: |
| COVID, Feb-Apr 2020 | −2.22% (worst −8.74%) | **0.00%** (flat throughout) |
| yen unwind, Jul-Sep 2024 | −13.44% (worst −13.87%) | **−6.06%** (worst −6.71%) |

Which is why it was worth taking to the test split rather than abandoning.

---

## 4. The warning, written before the test was run

The parameter neighbourhood was not a plateau, and this was on the record before
the held-out split was touched:

| parameter | dev SR | dev DD | val SR | val DD |
| --- | ---: | ---: | ---: | ---: |
| ma_days = 100 | 1.06 | −10.5 | 0.03 | −5.7 |
| ma_days = 150 | 0.94 | −9.6 | **−0.13** | −8.6 |
| **ma_days = 200** | 0.81 | −10.8 | 0.26 | −8.0 |
| ma_days = 250 | 0.72 | −13.1 | 0.33 | −8.3 |
| ma_days = 300 | 0.58 | −14.3 | **−0.01** | −9.0 |
| vol_days = 10 | 0.95 | −10.7 | 0.48 | −7.1 |
| **vol_days = 20** | 0.81 | −10.8 | 0.26 | −8.0 |
| vol_days = 60 | 0.75 | −11.2 | 0.25 | −7.9 |
| target_vol = 6% | 0.91 | −8.5 | 0.31 | −6.9 |
| **target_vol = 8%** | 0.81 | −10.8 | 0.26 | −8.0 |
| target_vol = 12% | 0.74 | −13.7 | 0.23 | −8.3 |
| max_weight = 0.75 | 0.78 | −9.6 | 0.24 | −6.3 |
| **max_weight = 1.0** | 0.81 | −10.8 | 0.26 | −8.0 |
| max_weight = 2.0 | 0.91 | −11.2 | 0.32 | −9.8 |
| band = 0 | 0.85 | −10.4 | 0.28 | −8.1 |
| **band = 0.10** | 0.81 | −10.8 | 0.26 | −8.0 |
| band = 0.20 | 0.80 | −10.3 | 0.16 | −8.5 |

**Two of 23 settings give a negative validation Sharpe**, and the two splits
disagree about which direction `ma_days` should move. The USTEC equivalent was
positive at all 27 settings in both splits, ranging 0.40 to 1.08. That contrast
was the reason to expect this one to be fragile, and it was.

---

## 5. The held-out split

| candidate | CAGR% | vol% | SR | maxDD% | Calmar | avg w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J0 buy and hold | **8.59** | 7.94 | **1.08** | −4.5 | 1.92 | 1.00 |
| J1 yen trend gate | 0.53 | 6.96 | 0.11 | −4.5 | 0.12 | 0.77 |
| **J2 — the selection** | **−0.11** | 6.22 | **0.01** | −4.2 | −0.03 | 0.68 |
| J3 J2 + equity risk gate | 0.06 | 6.05 | 0.04 | −4.2 | 0.01 | 0.64 |
| J4 equity risk gate only | 8.78 | 7.80 | 1.12 | −4.5 | 1.96 | 0.96 |
| J6 vol target, no gate | 7.54 | 7.11 | 1.06 | −4.1 | 1.84 | 0.89 |

By year: 2025 the strategy made **+0.06%** while holding made **+8.64%** at an
average weight of 0.44; 2026 −0.19% against +1.57%.

Two things to read here, and the second matters more than the first.

**The overlay bought nothing.** The test period had a −4.5% worst drawdown — the
calmest stretch in the corpus — so there was nothing for a defensive rule to
protect against, and it gave up the whole return standing aside.

**And the two candidates that top the test table are the two that failed
validation.** J4 scores a Sharpe of 1.12 on test and 0.09 on validation; J6
scores 1.06 and 0.05. Anyone selecting on the test split would pick J4, the
candidate that demonstrably provided zero protection in August 2024. That is the
circularity the split exists to prevent, and it is visible here in a single
table.

---

## 6. What was tried before the overlay

USDJPY got the same battery EURUSD did. No directional edge, same as everywhere:

| hypothesis | result | verdict |
| --- | --- | --- |
| Hour-of-day drift shape | dev vs validation correlation −0.06 (60m), +0.20 (30m) | noise |
| Intraday window selection | dev SR +0.31 to +0.73, **validation SR −2.13 to −3.71** | decisive failure |
| Minute-level autoregression | *momentum* here, not reversal — t = +19.8 at 120m, but edge peaks at **0.43x** a round turn | too small |
| Daily momentum, 5-250 day | 60-day keeps its sign (+2.86 dev, +1.86 val) but t < 1.6; the rest flip | no |
| Daily reversal | dev −1.73, validation +1.32 | sign flips |
| Day of week | Monday t = 2.61 in dev, −0.09 in validation | multiplicity |
| Volatility regime → return | dev +3.52 / +1.13 / +1.14, validation −0.21 / −4.01 / +0.61 | sign flips |
| Session opening-range breakout | previously evaluated: dev t 1.95, val t 0.10, test t −0.99, 2026 t **−2.20** | already rejected |

The minute-scale sign is worth one line: USDJPY shows **momentum** at 60-120
minute horizons where EURUSD and USTEC show reversal, which is what a trending
carry currency should look like. It is the largest such effect of the three
symbols measured — and at 0.43x of a round turn it is still not tradable.

---

## 7. What was kept

The refactor, which is reusable and is the durable output:

`risk_managed_long` now takes a **`SessionSpec`**, and ships two — `US_CASH`
(09:30–16:00 New York, signal and trade both at the cash open) and `FX_DAY` (the
FX day ends 17:00 New York; the trade happens at 08:00 the next morning). The
split matters because on FX the day boundary and a sane execution time are not
the same instant: 17:00 New York is the rollover, whose spread runs 1.5 bps
against a weekday mean near 0.015. Trading the close would hand back more than
the strategy makes.

Two bugs were fixed on the way and both are now asserted in tests:

- **The open must come from a bar that actually quoted.** The panel previously
  took the first bar in the whole session window, so a day that first quoted at
  10:30 was recorded as having opened there and called it 09:30. It is now an
  inner join on a ten-minute window at the open, and such days are dropped. This
  moved USTEC's dev CAGR by 0.05pp (one session); validation and test were
  unaffected, confirmed by one re-run.
- **A rolling mean over a window containing a null is null.** Computing USTEC's
  200-day average *after* joining it to the FX calendar ran the window over a
  series holed by US holidays, which silently switched the equity gate off on
  nearly every day. It is now computed on USTEC's own calendar and joined after.

---

## 8. Would the swap credit have saved it?

No. At the full measured basis of 1.47 bps a night, the credit is proportional to
the weight carried — and buy-and-hold carries more of it than the overlay does:

| test split, with full swap credit | CAGR (approx) |
| --- | ---: |
| buy and hold, weight 1.00 | 8.59% + 3.8% ≈ **12.4%** |
| J2, weight 0.68 | −0.11% + 2.6% ≈ **2.5%** |

The favourable financing asymmetry is real and it applies to *holding the pair*.
It does not rescue a rule whose contribution was to be out of the market during
the year it went up.

---

## 9. Where this leaves USDJPY

**Not deployable as a strategy.** What the corpus supports is narrower and
should be said plainly: over 2020–2026 a long USDJPY paid, the swap paid on top
of it, and a trend gate would have kept you out of the two unwinds — at the cost
of most of the return in the years it did not.

That is a description of one monetary-policy cycle, not an edge. The yen carry
trade is a documented risk premium with a documented crash profile, and this
sample contains one full cycle of it. Nothing here distinguishes "this rule
works" from "the yen fell for three years".

**The test split is spent for this family.** A future USDJPY study needs either
new data or a genuinely different hypothesis — not another configuration of this
one, and emphatically not J4 or J6 because they look good in §5.
