# The overnight drift - evaluation

Subject: Boyarchenko, Larsen and Whelan, *The Overnight Drift*, Federal Reserve
Bank of New York Staff Report 917 (Feb 2020, revised Aug 2022). S&P 500 futures
trade almost around the clock, but the return does not accrue around the clock:
the hour from **02:00 to 03:00 New York** - the European cash open - carries an
annualised 3.7% on its own, is positive in 20 of 23 years, and is the only hour
that survives a multiple-testing correction. The mechanism argued for is
inventory risk: selling pressure into the US close leaves dealers long, they
unwind into the first European liquidity, and they charge for the wait. The
prediction that follows is asymmetric - the drift should be biggest after a
close with **negative** order imbalance, and reversals after rallies should be
much weaker.

Method: implemented in `src/qlab/strategies/overnight_drift.py`, run on USTEC as
the S&P proxy with gold and two FX majors as a falsification set. **~1,010
sessions on dev (2020-2023), 383 on validation (2024 - H1 2025).** Gross returns
are mid to mid; net returns buy the ask and sell the bid at the measured spread
and then pay commission from the contract terms.

**Verdict: the paper's own honest conclusion replicates and its headline
statistic does not. The unconditional 02:00 hour is not distinguishable from
any other hour in 2020-2026 - it ranks 4th of 23 on dev and 12th of 23 on
validation, and nothing survives a correction on either split. The conditional
"buy the dip" version keeps a positive net Sharpe in both splits (0.47 dev,
0.60 validation) and its asymmetry has the paper's sign in both, but the paired
difference against its mirror image has p = 0.37. Rejected as a strategy;
retained as a documented, correctly-signed, statistically weak regularity. The
test split was not spent.**

---

## 0. What this corpus can and cannot say about this paper

**Different instrument.** The paper trades ES; this corpus holds USTEC, a
Nasdaq-100 CFD on the same 23-hour cycle driven by the same US equity flow.
Nothing in the mechanism is S&P-specific, but a failure here is ambiguous
between "the effect is gone" and "the effect is not in this instrument", and
that ambiguity is live wherever it matters below.

**Different era - and it is the era *after* publication.** The paper's sample
ends December 2020; this corpus starts January 2020. The overlap is eleven
months, so this is very nearly a pure out-of-sample test of a published result,
run across the period in which NightShares launched two ETFs (June 2022, wound
down 2023) explicitly to harvest this pattern. A decayed effect was the most
likely outcome before any number was computed, and it is what the numbers show.

**Order imbalance is a proxy, and there is a no-proxy control.** The paper signs
*trade* volume by aggressor. This feed carries quotes and no trades, so
`closing_imbalance` builds the nearest stand-in - each 1-minute bar in the
15:00-16:00 window signed by its own return and weighted by its quote-update
count. Because a dead proxy and a dead effect look identical, every conditional
result is also run on the **closing-hour return itself**, which needs no proxy
at all. Both are reported.

**No risk-free rate.** Sharpe ratios here are mean over standard deviation, with
no rate subtracted; the project holds quotes and nothing else. Over 2022-2025
that subtraction would move the overnight numbers down by a few tenths, and the
comparison against the paper's figures should be read with that in mind.

**No swap.** None of the windows spans the 21:00 UTC financing point, so no
strategy in this study carries an overnight charge. This is the one place the
CFD is *cleaner* than a futures replication, which would need a roll.

---

## 1. The hour is not special any more

Every clock hour of USTEC, mean return in basis points, with Bonferroni and
Benjamini-Hochberg corrections over the 23 hypotheses the clock face contains.
Dev:

| hour | n | mean bps | ann % | t | p | p Bonf | p BH |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 1010 | +1.506 | +3.80 | +2.32 | 0.020 | 0.465 | 0.254 |
| 01 | 1009 | +0.717 | +1.81 | +1.23 | 0.219 | 1.000 | 0.719 |
| **02** | 1009 | **+0.989** | **+2.49** | **+1.36** | 0.175 | 1.000 | 0.719 |
| 07 | 1008 | -1.526 | -3.85 | -1.84 | 0.065 | 1.000 | 0.375 |
| 11 | 1008 | +2.980 | +7.51 | +2.09 | 0.037 | 0.844 | 0.281 |
| 14 | 979 | +3.020 | +7.61 | +2.29 | 0.022 | 0.509 | 0.254 |

(Full table in `overnight_drift.json`; the rows above are the extremes plus the
drift hour.)

The drift hour's **point estimate replicates**: +2.49% annualised against the
paper's +3.7%, same sign, same order of magnitude, on an instrument and in an
era the paper never saw. What does not replicate is its *distinctiveness*. It
ranks **4th of 23 by t-statistic on dev and 12th of 23 on validation**, and on
neither split does any hour survive Bonferroni or Benjamini-Hochberg. The
paper's central empirical claim is not "2:00 is positive" - it is "2:00 is the
only hour that survives correction", and that claim fails here.

Four years is the reason it *can* fail without contradicting the paper. The
paper has 23 years and a t of roughly 5; a four-year slice of the same effect
would be expected to produce a t near 1.4, which is precisely what dev gives.
This study cannot distinguish "the effect decayed" from "the effect is intact
and four years is too short to see it". It can only say that in 2020-2026,
looking at this instrument, the hour is not identifiable.

---

## 2. The paper's Table IX, side by side

Annualised, USTEC, after crossing the spread **and** paying commission - the
strictest column - with the paper's own post-cost figure alongside.

| strategy | dev ann % | dev Sharpe | val ann % | val Sharpe | paper (post-cost) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CTC (passive) | +20.15 | +0.73 | +21.99 | +0.96 | +0.42 |
| CTO (overnight leg) | +10.09 | +0.64 | +19.49 | +1.35 | -0.04 |
| OTC (daytime leg) | +11.25 | +0.54 | -0.91 | -0.05 | -0.06 |
| OD (02:00-03:00) | **-0.27** | **-0.07** | **-1.45** | **-0.56** | **-0.54** |
| OD+ (01:30-03:30) | +1.60 | +0.27 | +1.72 | +0.40 | +0.26 |
| **BtD** (OD+ after RSV < 0) | **+2.05** | **+0.47** | **+2.00** | **+0.60** | **+1.10** |
| BtD(ret) (after a down close) | +1.62 | +0.39 | +2.45 | +0.70 | - |
| Rally (OD+ after RSV > 0) | -0.07 | -0.02 | -0.51 | -0.20 | - |

Read down the OD row first, because it is the paper's own headline and the
replication is nearly exact: **the drift hour does not survive its own bid-ask
spread.** Gross it earns +2.64% a year on dev; the spread takes two thirds and
commission takes the rest. The paper gets a post-cost Sharpe of -0.54 and this
study gets -0.07 on dev and -0.56 on validation. That is not a failed
replication - it *is* the paper's result, and the paper says so: market makers
position the book so the contrarian trade does not pay.

The OD+ row is the same story with a wider window and lands within 0.15 of the
paper's post-cost Sharpe on both splits. And BtD, the only strategy the paper
finds tradeable, is the best strategy here too, on both splits, in the same
order relative to OD+ - it just earns a third to a half of what the paper
reports.

The two rows without a paper column are this study's additions. `BtD(ret)`
replaces the imbalance proxy with the closing-hour return and does at least as
well, which matters: it means the conditional result is not an artefact of the
proxy. `Rally` is the mirror image and is the subject of the next section.

One warning about the CTO row on validation: +19.5% a year at a Sharpe of 1.35
is the overnight leg of a Nasdaq bull market with the daytime leg contributing
nothing, over 18 months. It is the largest number in the table and the least
robust thing in it.

---

## 3. The asymmetry has the right sign and no significance

The paper's mechanism lives or dies here. BtD and Rally hold **the same window
at the same cost on complementary halves of the same calendar**; the only
difference is the sign of the previous close's imbalance. If the two agree, the
inventory story is not what is happening.

| | dev | validation |
| --- | ---: | ---: |
| BtD, days held | 492 / 1013 | 195 / 383 |
| BtD Sharpe (net) | +0.47 | +0.60 |
| BtD 95% block-bootstrap | [-0.57, +1.37] | [-1.11, +1.87] |
| Rally, days held | 484 / 1013 | 170 / 383 |
| Rally Sharpe (net) | -0.02 | -0.20 |
| Rally 95% block-bootstrap | [-1.10, +0.89] | [-2.03, +1.23] |
| **BtD - Rally, bps/day (paired)** | **+0.841** | **+0.999** |
| paired 95% | [-0.94, +2.63] | [-1.18, +3.34] |
| paired p | 0.367 | 0.377 |

The sign is right twice, the magnitude is almost identical across two
independent periods (+0.84 and +1.00 bps a day), and the interval covers zero
both times with p near 0.37. This is the most informative row in the study and
it is genuinely two-sided: an effect that reproduces its magnitude out of sample
is not usually nothing, and a p of 0.37 is not evidence of anything. Both
statements are true and neither should be dropped.

---

## 4. The sorts and the predictability regression

**Sorted on the previous close** (OD+ net, by quintile). On dev, the no-proxy
version shows the paper's shape - the most negative closes are followed by the
biggest drift:

| quintile | signal | dev mean bps | dev t |
| ---: | ---: | ---: | ---: |
| 1 (biggest selloff) | -69.6 bps | **+7.49** | +2.12 |
| 2 | -17.1 | -2.22 | -1.19 |
| 3 | -1.0 | -1.38 | -0.71 |
| 4 | +16.2 | -1.67 | -0.82 |
| 5 (biggest rally) | +62.6 | +1.92 | +0.56 |

That is the paper's prediction in two respects at once: the extreme selloff
quintile is the only one that pays, and the middle quintile - where dealers end
the day roughly flat and, on the paper's argument, should require no liquidity
premium - is indistinguishable from zero. On validation the same sort is flat
(quintile 1: +0.28 bps, t = +0.06), so the shape does not carry.

The **RSV proxy** sort is non-monotone on both splits. Given that the no-proxy
version works on dev and the proxy does not, the reasonable inference is that
signing quote revisions is a poor stand-in for signing trades by aggressor -
which is a limitation of this feed, not a finding about the paper.

**Predictability (the paper's Table VII).** Regressing each hour's return on the
previous close's imbalance with Newey-West errors, the paper finds a significant
negative loading at the London and Frankfurt open and nowhere else. On dev the
02:00 hour has the paper's sign (beta -7.67) at t = -1.36; on validation it is
zero (beta +0.51, t = +0.08). No hour on either split is significant after
accounting for the 23 tested. The mechanism evidence, like the effect, is
directionally intact and statistically absent.

---

## 5. Falsification, and one uncomfortable row

The same 02:00 hour on the other three instruments. An inventory story about US
equity order flow predicts nothing for EURUSD:

| | USTEC | XAUUSD | EURUSD | USDJPY |
| --- | ---: | ---: | ---: | ---: |
| dev ann % | +2.49 | **+2.50** | -1.37 | +0.64 |
| dev t | +1.36 | +1.67 | -1.59 | +0.70 |
| val ann % | +0.67 | +0.85 | -0.08 | +0.12 |
| val t | +0.32 | +0.37 | -0.07 | +0.06 |

The FX pairs behave as the mechanism requires - nothing at 02:00, on both
splits. **Gold does not.** It shows the same drift as USTEC on dev, slightly
larger and with a slightly better t.

Two readings, and this study cannot choose between them. Gold is a
dollar-denominated risk asset whose European-hours liquidity arrives at the same
moment, so an inventory story could plausibly cover it; or the 02:00 effect on
both is a shared response to the European open that has nothing to do with US
equity dealers' books. What can be said is that the falsification test **half
passes**: it excludes the FX majors, which is the outcome that would have been
most damaging had it gone the other way, and it does not cleanly isolate US
equity flow.

---

## 6. What this does not say

- **It does not contradict the paper's tradeability conclusion - it confirms
  it.** The paper's own summary is that the drift is "not easily profitable",
  and every number here agrees: OD is negative after costs on both splits, in
  line with the paper's own post-cost Sharpe of -0.54.
- **It does not show the drift has disappeared.** Four years cannot resolve a
  3.7%-a-year effect. The point estimate on dev is two thirds of the paper's,
  with the sign right and the t where a four-year slice of the paper's own
  effect would put it.
- **It does not clear BtD for deployment.** +2% a year at 3.4% volatility, with
  a Sharpe interval from -0.6 to +1.4, requires overnight leveraged index
  exposure to earn a return that a money market fund matched over the same
  period. The consistency across splits is interesting; the economics are not.
- **The test split is unspent.** Nothing here reached the bar of a live
  candidate.

---

## 7. If you want to keep pulling this thread

1. **Get real signed volume.** The single largest limitation is that the paper's
   conditioning variable cannot be measured here. The no-proxy control works on
   dev and the proxy does not, which is a specific, fixable data problem rather
   than an inference problem.
2. **Pool the instruments.** USTEC and gold both show the 02:00 hour on dev.
   Testing them as a two-asset panel with a common intercept has more power than
   testing either alone, and would sharpen the falsification in section 5 into
   an actual answer.
3. **Condition on volatility as well as on imbalance.** The paper's double sort
   on imbalance and the VIX is the sharpest version of its mechanism, and
   `realized_vol_proxy` in `qlab.strategies.overnight_reversal` already supplies
   a labelled VIX stand-in built from the index's own realised volatility.

---

## 8. Reproducing

```bash
python scripts/backtests/backtest_overnight_drift.py --splits dev validation
```

Under a minute; writes `reports/strategies/overnight_drift.json`. Tests:
`pytest tests/test_overnight_drift.py`.
