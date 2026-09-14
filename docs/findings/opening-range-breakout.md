# The 5-minute opening range breakout - evaluation

Subject: Zarattini, Barbon and Aziz (2024), *A Profitable Day Trading Strategy
For The U.S. Equity Market* (SSRN 4729284). Take the first 5 minutes of the US
session as a range; let the sign of that candle pick the side and only that
side; enter on a stop order at the range edge; stop out at **10% of the 14-day
ATR**; hold the rest to the bell. Applied to ~7,000 US stocks over 2016-2023
this returns 29% - worse than the index. The paper's advertised version adds a
**Stocks in Play** filter, keeping only names whose opening-range volume beats
its own 14-day average and trading the top 20 by that ratio, and returns 1,600%
at a Sharpe of 2.81.

Method: implemented in `src/qlab/strategies/opening_range.py`, resolved against
the **tick tape** rather than bars, on USTEC as the primary instrument with
gold and two FX majors as breadth checks. **962 sessions on dev (2020-2023),
368 on validation (2024 - H1 2025).** Entries and exits cross a real bid and a
real ask; commission comes from the contract terms; slippage from the measured
cost model.

**Verdict: the geometry has an edge on dev and does not carry to validation.
The paper's exact rule earns +0.198 R per trade on dev (t = +2.12, bootstrap
p = 0.023) and +0.077 R on validation (t = +0.50, p = 0.62). The Stocks in Play
filter - the paper's actual contribution - does not transfer to a single index
and destroys the edge when applied. The held-out test split was not spent.**

The one result that does replicate cleanly is an *ordering*, in both splits and
in two independent directions:

| ordering | dev | validation |
| --- | --- | --- |
| paper's sign > both sides > opposite side | +0.198 / +0.144 / +0.029 R | +0.077 / +0.037 / **-0.091** R |
| USTEC > gold > EURUSD > USDJPY | +0.198 / +0.096 / +0.002 / -0.051 R | +0.077 / +0.040 / -0.140 / -0.150 R |

Both orderings are what the paper's mechanism predicts, and neither is
individually significant. That combination - right shape, no power - is the
honest summary of this whole study.

---

## 0. What this corpus can and cannot say about this paper

**The geometry transfers; the cross-section does not.** Rules 1-4 - range,
directional filter, ATR stop, hold to the bell - are single-instrument rules and
are implemented exactly. "Trade the top 20 names by relative volume" needs a
universe of names to rank and there is none here; with one instrument a day the
ranking is a tautology. The part of the filter that carries the mechanism -
`RelVol >= 1`, a per-name per-day quantity - **is** implementable and is tested,
as is the bucket sort that is the paper's Figure 4.

**Volume is quote updates, not shares.** The feed carries no size. The stand-in
for opening-range volume is the **tick count**: how many quote revisions the
broker published in the window. Because RelVol is a ratio to the same
instrument's own trailing mean, the units cancel, which is what makes the proxy
usable at all. It is still a proxy and is labelled as one everywhere below.

**The universe screens are not applicable.** Price > $5, 14-day volume > 1M
shares, ATR > $0.50 exist to throw out penny stocks. USTEC and gold pass all
three trivially, and no screening is performed.

**One instrument is not 7,000.** The paper's Sharpe of 2.81 comes mostly from
diversification: twenty roughly independent bets a day instead of one. Nothing
here can produce that number and nothing here should be read as trying to. What
is testable is whether the **per-trade expectancy** the portfolio is built from
exists at all, and that is what every number below measures.

---

## 1. Why this is resolved on ticks, not bars

The stop is 10% of a 14-day ATR. On USTEC over dev that is **25.8 index points**
against an opening range averaging **50 points** and 1-minute bars whose range
in the first hour routinely exceeds both. So the bar that contains the entry
very often contains the stop as well, and a bar-level engine has to *assume*
which came first. That assumption is worth more than the edge being measured:
resolving every such bar in the trade's favour turns a losing strategy into a
winning one on this geometry.

Every entry and exit here is therefore resolved tick by tick - a buy stop
triggers on the ask, a long stops out on the bid, and a tick that reaches the
stop and the target at once is scored as a stop-out, because a tick carries a
bid and an ask and not a path.

| | dev | validation |
| --- | ---: | ---: |
| sessions | 962 | 368 |
| order filled | 910 (94.6%) | 350 (95.1%) |
| ATR(14) mean | 257.9 pts | 331.6 pts |
| 1R = 10% ATR | 25.8 pts | 32.9 pts |
| opening range width | 50.0 pts (0.19 ATR) | 57.3 pts (0.17 ATR) |
| round turn | 0.079 R | 0.072 R |
| median minutes 09:35 to fill | 0.6 | 0.8 |

Two things to read off that table. The order fills on **95% of sessions**, which
means the "breakout" is not a rare event on an index - the range is a fifth of a
daily ATR and price leaves it almost immediately. And the round turn is **8% of
one unit of risk**, which is cheap: this geometry is not cost-bound, so
everything that follows is about the signal and not about execution.

---

## 2. The rule exactly as written

| | dev | validation |
| --- | ---: | ---: |
| trades | 910 | 350 |
| mean R, net | **+0.198** | **+0.077** |
| t (Newey-West) | +2.12 | +0.50 |
| block-bootstrap 95% | [+0.033, +0.372] | [-0.212, +0.380] |
| bootstrap p | 0.023 | 0.620 |
| hit rate | 18.8% | 18.0% |
| mean winner | +5.59 R | +5.14 R |
| mean loser | -1.05 R | -1.03 R |
| total | +180.1 R | +27.0 R |
| max drawdown | 38.3 R | 63.9 R |

The **shape** is exactly the paper's, and this is worth saying plainly because
it is the part that reproduces: fewer than one trade in five wins, the average
loss is a hair worse than the planned 1R because a stop is a market order, and
the whole return comes from a small number of trades that never get stopped and
run to the bell. On dev, 175 trades exited at the bell for +5.46 R each against
735 stop-outs at -1.06 R. This is a convexity trade, not a prediction: it buys a
cheap option on an intraday trend.

Sized the paper's way - a stop-out costs 1% of equity - dev compounds **+311%
over four years (47.9% a year, Sharpe 1.05) with a 32.9% drawdown**. That number
should be read as a description of the R curve, not as a plan; at 1% risk per
trade the drawdown is already a third of the account, and validation's R curve
gives +14.9% total with a **48.2% drawdown**, which is a different account
entirely.

Year by year on dev the edge is present in every year and dominated by none -
2020 (+0.114 R) is if anything the *weakest*, which rules out the obvious
objection that this is a COVID artefact:

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 H1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +0.114 | +0.276 | +0.116 | +0.276 | +0.088 | +0.055 |

Read across, that is a series that decays rather than breaks. It is consistent
with a real effect being competed away after a widely-read paper, and equally
consistent with four years of noise around zero. This study cannot separate
those two, and says so rather than picking the flattering one.

---

## 3. The exit surface: a plateau, and the paper is standing on it

Six stop widths by seven exits, net of everything, mean R (dev):

| stop | EOD | 0.5R | 1R | 2R | 3R | 5R | 10R |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | +0.183 | -0.052 | -0.075 | -0.042 | -0.083 | -0.072 | +0.088 |
| **0.10** | **+0.198** | -0.046 | -0.008 | -0.021 | +0.047 | +0.109 | +0.193 |
| 0.15 | +0.072 | -0.051 | -0.057 | -0.006 | +0.025 | +0.080 | +0.075 |
| 0.25 | +0.023 | -0.042 | -0.033 | -0.004 | +0.032 | +0.024 | +0.024 |
| 0.50 | -0.011 | -0.046 | -0.022 | -0.009 | -0.014 | -0.011 | -0.011 |
| 1.00 | -0.009 | -0.017 | -0.007 | -0.009 | -0.009 | -0.009 | -0.009 |

Three readings, in order of how much they matter.

**No profit target helps.** Every column left of 10R is worse than holding to
the bell, and the tighter the target the worse it gets - 0.5R is negative at
every stop width. That is Wu et al. (2020)'s result, which the paper cites, and
it falls straight out of the return shape in section 2: capping the winners at
2R while the losers stay at -1R throws away the entire distribution.

**The stop is not a knife edge in the direction that matters.** Going from 0.10
to 0.05 costs little; going wider decays smoothly to zero by 0.50 ATR and stays
there. The decay has a mechanical cause: as the stop widens, R grows, the
strategy converges on "hold the session" and the per-R edge is divided away. The
sharp step is only between 0.10 and 0.15, which is the one place a
0.10-vs-0.15 coin flip would have mattered.

**The paper is not standing on a fitted cell.** 0.10/EOD is fixed in the paper
in advance of this data; 42 cells were searched here and none of them was. It
lands at the maximum of the net surface, and on validation it lands mid-table
(+0.077 against a best of +0.126 at 10R and +0.091 at a 0.25 stop) - which is
what an unfitted parameter looks like on new data.

---

## 4. Does the opening candle's sign do any work?

This is the paper's one genuinely directional claim - a bearish first candle
means you may only short - and it is the claim this corpus is best placed to
test, because it can be tested against its own null three different ways.

| variant | dev mean R | dev t | validation mean R | validation t |
| --- | ---: | ---: | ---: | ---: |
| paper: sign of the candle | +0.198 | +2.12 | +0.077 | +0.50 |
| control: take both sides | +0.144 | +1.52 | +0.037 | +0.26 |
| contra: take the opposite side | +0.029 | +0.30 | **-0.091** | -0.54 |
| placebo: shuffled sign, 60 draws | +0.133 (mean) | - | -0.007 (mean) | - |

The ordering `paper > both > contra` holds on both splits, and on validation the
contra leg goes negative - the sign is doing *something*, consistently, in the
direction the paper says.

But it does not survive its own placebo. Shuffling the sign across sessions
while keeping every other piece of the geometry - the same calendar, the same
ranges, the same ATR, the same stop - produces a mean of **+0.133 R** on dev,
and the real signal at +0.198 R is beaten by **11 of 60 shuffles (empirical
p = 0.197)**. On validation, 10 of 60 (p = 0.180). Twice, the paper's rule lands
in the upper quartile of its own null and not outside it.

The reason the placebo scores so well is the finding: **most of the edge is in
the geometry, not the direction.** A tight ATR stop with an uncapped hold to the
bell is profitable on this instrument taken *at random*, because 18% of sessions
trend far enough to pay for the other 82%. The paper's directional rule adds
perhaps a third of the total and cannot be distinguished from zero at this
sample size.

---

## 5. Stocks in Play does not transfer, for a reason worth stating

The paper's Figure 4 sorts trades by relative volume and finds -0.02 R below
100%, +0.08 R above it, and +0.38 R above 3,000%. Here, dev:

| RelVol bucket | n | mean R | t |
| --- | ---: | ---: | ---: |
| 0.5 - 1.0 | 503 | **+0.296** | +2.27 |
| 1.0 - 1.5 | 389 | +0.014 | +0.09 |
| 1.5 - 2.0 | 15 | +1.513 | +1.51 |
| 2.0 - 3.0 | 2 | +2.006 | +0.65 |

The sort is **backwards** in the range where the data lives, and the buckets
where the paper's effect is supposed to be strong are two and fifteen trades
wide. Applying the filter as the paper specifies cuts dev from +0.198 R to
+0.076 R and validation is flat (+0.077 to +0.078).

This is a failure the mechanism predicts. Relative volume identifies *Stocks in
Play* - a name with a fundamental catalyst that day, an earnings surprise or an
FDA decision, which is why the paper's cross-section reaches 30x. An index has
no idiosyncratic catalyst. Its opening-range volume never reaches 3x, let alone
30x; the highest bucket populated here is 2-3x with two trades in it, and what
"high relative volume" means on an index is a macro event - CPI, a Fed day - on
which everything moves together. The paper's filter is a *stock-selection*
device, and there is nothing to select. It should not be read as a refutation of
the paper's central result; it should be read as this corpus being the wrong
place to test it.

---

## 6. Range length and breadth

**Range length** (the paper's Section 5, which finds 5 minutes best):

| | 5m | 15m | 30m | 60m |
| --- | ---: | ---: | ---: | ---: |
| dev mean R | +0.198 | +0.223 | +0.123 | +0.125 |
| validation mean R | +0.077 | +0.202 | -0.070 | -0.083 |

The short ranges beat the long ones on both splits, which is the paper's
finding. Fifteen minutes edges five on both, which is not - but the gap is well
inside the noise, and picking it here after seeing both splits would be exactly
the fitting this report is trying to avoid.

**Breadth** - the same 09:30 New York rule on the other three instruments:

| | USTEC | XAUUSD | EURUSD | USDJPY |
| --- | ---: | ---: | ---: | ---: |
| dev mean R | +0.198 | +0.096 | +0.002 | -0.051 |
| validation mean R | +0.077 | +0.040 | -0.140 | -0.150 |

This is the most encouraging table in the study and it is worth being careful
about why. The effect is **monotone in how much the instrument is driven by the
US equity open**: strongest on the Nasdaq index, half as strong on gold, absent
on EURUSD, negative on USDJPY. That is not what a generic breakout artefact
looks like - a bar-resolution bug or a stop-placement error would show up on all
four. It is what a real US-equity-open effect looks like, seen through
instruments with varying exposure to it, and it holds in both splits.

It is also not significant on any single instrument, and four instruments is not
a cross-section.

---

## 7. What this does not say

- **It does not refute the paper.** The paper's result is a portfolio of twenty
  stock-specific bets a day, and the number it advertises comes largely from
  diversification and from a stock-selection filter that has no meaning on an
  index. This study tested the per-trade geometry that portfolio is assembled
  from, on one index, and found it positive on dev and unconfirmed on
  validation.
- **It does not clear the geometry for deployment.** +0.077 R per trade on
  validation, with a bootstrap interval from -0.21 to +0.38 and a 48% peak-to-
  trough in R units, is not a tradeable estimate. Nothing here should be sized.
- **It says nothing about US equities.** No stock was tested. Whether the effect
  survives 2024-2026 in the paper's own universe is answerable only with an
  equity cross-section, and that is the single highest-value thing this study
  could not do.
- **The test split is unspent.** Under the rule that `test` is spent only on a
  live candidate, validation's t of +0.50 is not one. The split remains
  available.

---

## 8. If you want to keep pulling this thread

1. **The geometry without the signal is the more interesting object.** The
   placebo result says a tight-stop, uncapped-hold structure on the US open
   makes +0.13 R at random. That is a statement about USTEC's intraday return
   distribution - the convexity of the 09:35-to-bell move - not about opening
   ranges. It could be measured directly, with far more power, as a forward
   return and MFE/MAE study rather than as a strategy.
2. **The 15-minute range, pre-registered.** It beat 5 minutes on both splits.
   Fixing it now and testing once on `test` is a legitimate single-shot
   experiment; picking it after this report is not. *Run 2026-09-11:
   +0.086 R at t = +0.57 on test, inconclusive - see [orb15.md](orb15.md). The
   test split is spent for this family.*
3. **An equity cross-section.** Everything the paper's headline depends on -
   twenty names, relative volume reaching 30x, diversification - needs one.
   Until there is one, this is the only part of the paper this corpus can speak
   to.

---

## 9. Reproducing

```bash
python scripts/backtests/backtest_opening_range.py --splits dev validation --placebo-draws 60
```

Roughly four minutes; writes `reports/strategies/opening_range.json` and one
trade tape per split under `reports/strategies/opening_range_trades/`. Tests:
`pytest tests/test_opening_range.py`.
