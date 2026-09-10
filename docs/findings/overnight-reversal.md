# The overnight-intraday reversal family - evaluation

Subject: Liu, Liu, Wang, Zhou and Zhu (2025). Split the daily return into an
overnight leg (previous close to today's open) and an intraday leg (today's open
to today's close), and the traditional short-horizon reversal turns out to be
carried by one component: sort the cross-section on **yesterday's overnight
return**, hold **today's intraday return**, contrarian - "CO-OC". Reported
profitable in equity index, rate, commodity and currency futures alike, and
conditioned not on the VIX or on sentiment but on the **cross-sectional
dispersion of overnight returns**.

Method: implemented in `src/qlab/strategies/overnight_reversal.py`, run on the
four instruments this corpus holds - EURUSD, USDJPY, XAUUSD, USTEC - as a single
zero-investment cross-section. **976 complete sessions on dev, 366 on
validation.** All four variants of the paper's Table 1 plus the two it puts in
its appendix, under three weighting schemes, at daily and weekly frequency,
gross and net of a measured round-turn cost.

**Verdict: the paper's primary strategy does not replicate here, and the
component it is supposed to dominate is the only one with a pulse. Rejected on
dev; the held-out test split was not spent.** CO-OC nets -1.05 bps a day
(t = -0.34) and is indistinguishable from its own shuffled placebo - five of ten
placebo seeds beat it. The traditional CC-CC reversal it was meant to supersede
nets +6.60 bps (t = +1.76), which is the largest number in the study and still
does not survive a correction for the four hypotheses that produced it. The
mechanism claim fails in a specific and informative way: overnight dispersion
does predict the strategy's **volatility** (t = +2.92) and does not predict its
**return** (t = -0.17).

---

## 0. What this corpus can and cannot say about this paper

Three limits, stated before any number, because two of them are large.

**Breadth.** N = 4, against the paper's dozens per asset class. A demeaned
zero-investment portfolio of four assets is a legitimate portfolio, but it has
essentially no diversification and its standard errors are correspondingly wide.
This study can say "not found here"; it cannot say "not there". Where the
distinction matters below, it is drawn.

**There is no exchange close.** These are spot CFDs quoting almost continuously,
so the literal close-to-open gap is a tick or two on EURUSD. The paper's futures
have the same property - the CME trades nearly 23 hours - and the resolution is
the exchanges' own: the session is the **regular trading hours of the matching
futures contract**, and everything outside it is "overnight".

| instrument | contract | session (New York) |
| --- | --- | ---: |
| USTEC | NQ | 09:30 - 16:00 |
| XAUUSD | GC | 08:20 - 13:30 |
| EURUSD | 6E | 08:20 - 15:00 |
| USDJPY | 6J | 08:20 - 15:00 |

**No external data.** VIX, Baker-Wurgler sentiment, the Fama-French library, the
Asness value and momentum factors and an announcement calendar are all outside a
project that holds quotes and nothing else. The paper's risk-adjustment
regressions (their Table 4) and the global two-factor pricing test with its GRS
statistic (Table 5) are therefore **not implemented and not approximated** - a
GRS test on four instruments would be arithmetic without content. What *is*
implemented is everything that needs only the panel's own returns, which
includes the entire mechanism section, and a realised-volatility proxy for the
VIX that is labelled a proxy at every appearance.

---

## 1. The weighting scheme is not a detail here

The paper's construction is `w_i = -(1/N)(s_i - mean(s))`, applied *within* an
asset class where volatilities are comparable. This cross-section mixes gold and
USTEC with EURUSD, whose overnight standard deviation is a third of USTEC's:

| | EURUSD | USDJPY | USTEC | XAUUSD |
| --- | ---: | ---: | ---: | ---: |
| overnight return sd, bps | 34.8 | 43.8 | 107.7 | 66.5 |
| intraday return sd, bps | 36.8 | 39.2 | 130.6 | 69.6 |

Under the literal demeaned construction the portfolio is therefore almost
entirely a bet on whichever of gold and USTEC moved most overnight, and EURUSD
carries a weight indistinguishable from zero. Three weightings are run and all
three reported: `demean` (the paper's baseline, transcribed literally), `rank`
(the paper's own robustness variant, scale-free, and the honest primary
specification for a mixed panel), and `vol_scaled` (demeaned after dividing by
each asset's own trailing volatility - not in the paper, and what its
within-class construction is implicitly doing).

The verdict does not turn on the choice. CO-OC is flat to negative under all
three.

## 2. The core table

Mean daily return in bps of capital, Newey-West t-statistics, dev split.

| weighting | variant | gross | t | net | t | Sharpe | turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| demean | CC-CC | +8.76 | +2.09 | +6.83 | +1.63 | +0.77 | 3.59 |
| demean | OO-OO | +4.58 | +1.14 | +2.67 | +0.66 | +0.32 | 3.60 |
| demean | OC-OC | +7.94 | +2.38 | +5.82 | +1.75 | +0.84 | 2.00 |
| demean | **CO-OC** | **+0.77** | **+0.23** | **-1.29** | **-0.39** | **-0.20** | 2.00 |
| rank | CC-CC | +7.88 | +2.10 | +6.60 | +1.76 | +0.83 | 2.48 |
| rank | OO-OO | +1.82 | +0.52 | +0.56 | +0.16 | +0.08 | 2.47 |
| rank | OC-OC | +5.32 | +1.83 | +3.26 | +1.12 | +0.54 | 2.00 |
| rank | **CO-OC** | **+0.98** | **+0.32** | **-1.05** | **-0.34** | **-0.18** | 2.00 |
| vol_scaled | CC-CC | +3.87 | +1.17 | +2.21 | +0.67 | +0.32 | 3.59 |
| vol_scaled | OO-OO | +4.09 | +1.22 | +2.42 | +0.72 | +0.37 | 3.64 |
| vol_scaled | OC-OC | +5.93 | +2.39 | +4.03 | +1.63 | +0.75 | 2.00 |
| vol_scaled | **CO-OC** | **-1.40** | **-0.49** | **-3.27** | **-1.15** | **-0.63** | 2.00 |

The paper's central ordering is **inverted**. CO-OC is the weakest of the four
in every weighting; CC-CC, the strategy it is supposed to explain away, is the
strongest. And the gross-to-net column is doing real work, which is the subject
of section 3.

The two appendix variants, which the paper itself warns against because both
hold the overnight window and would have to be executed into the thinnest hours
of the day, are no better: CO-CO nets +1.48 bps (t = +0.63) and OC-CO nets
-0.22 bps (t = -0.09), before any allowance for the fact that the quoted
overnight spread is not the spread you would get.

CO-OC's correlation with the other three is low - +0.34 with CC-CC, +0.005 with
OO-OO, +0.05 with OC-OC - so it is at least measuring something distinct. It is
simply measuring something that does not pay.

## 3. Half the ranking is the cost asymmetry, and the paper does not charge it

CO-OC and OC-OC are **flat outside the session**. They enter at the open and
exit at the close, every single day, which is a full round turn per day on the
whole gross position. CC-CC and OO-OO **hold through**, and pay only for the
change in weight - one side of a round turn per unit traded.

Measured round-turn cost at the session open, in bps of price, dev split:

| EURUSD | USDJPY | USTEC | XAUUSD |
| ---: | ---: | ---: | ---: |
| 0.61 | 0.67 | 1.45 | 1.19 |

At a gross exposure of 1.0 and 2x-of-capital gearing that is roughly 2 bps a day
of drag on the intraday variants against roughly 1.3 bps on the continuous ones,
despite the continuous ones turning over *more* notional. A study that charged
both the same way - or charged neither - would report the four variants in a
different order and would not say so.

The turnover column is the thing to sit with: CO-OC runs **504x notional a
year**. That is a strategy whose entire annual gross edge is spent about twice
over before anything else happens.

## 4. Both legs are silent

Return per dollar committed to each side, rank weighting, dev. If the effect
were price pressure being unwound, both legs should contribute.

| variant | long | t | short | t |
| --- | ---: | ---: | ---: | ---: |
| CC-CC | +7.38 | +2.43 | +0.68 | +0.24 |
| OO-OO | +4.83 | +1.57 | -2.17 | -0.86 |
| OC-OC | +4.65 | +2.19 | +1.01 | +0.49 |
| CO-OC | +1.89 | +0.88 | -1.00 | -0.46 |

(The short leg is quoted as the return of the shorted basket, so a profitable
short is a negative number.) Whatever CC-CC and OC-OC have is in the long leg
only, which on a four-asset panel over 2020-2023 is not a comfortable place for
an edge to live - see section 7.

## 5. The mechanism fails, and it fails informatively

This is the part of the paper that is fully implementable here, and the result
is sharper than the return tables.

**Dispersion does not predict the return.** Regressing CO-OC's net daily return
on lagged overnight dispersion and its own lag:

```
term                     coef        t        p
const                -1.81385    -0.34   0.7311
dispersion[t-1]      -0.92864    -0.17   0.8632
self[t-1]           0.0388307    +1.25   0.2102
R2 0.0015   n 915   NW lags 6
```

The paper predicts a positive, dominant, asset-class-general coefficient. What
is here is zero. Adding the realised-volatility proxy changes nothing
(dispersion -1.04, t = -0.19; proxy +0.19, t = +0.52).

For OC-OC the coefficient is not zero - it is **the wrong sign and significant**
(-12.28, t = -2.19). High-dispersion days are followed by *worse* intraday
reversal returns, not better.

**Dispersion does predict the volatility.** The paper's own two-step conditional
Sharpe procedure separates these, and here it separates them cleanly. With
kappa = 1.281:

| step | regressor | coef | t |
| --- | --- | ---: | ---: |
| 1. conditional volatility | dispersion[t-1] | +47.69 | **+2.92** |
| 2. risk-adjusted return | dispersion[t-1] | -0.0132 | -0.20 |

So dispersion is a genuine forecaster of how violent tomorrow's session will be,
and carries no information about which way it goes. That is exactly the failure
mode the two-step design exists to expose, and it is the single most useful
result in this study: a specification that only ran step 2's numerator would
have reported a "signal" that is entirely a volatility forecast.

**The conditioning splits agree.** Mean net return in bps above and below each
conditioner's median, conditioner lagged:

| variant | conditioner | high | t | low | t | diff | t |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CO-OC | dispersion | -2.93 | -0.64 | -2.53 | -0.64 | -0.40 | -0.07 |
| CO-OC | vol proxy | +5.04 | +1.03 | -6.28 | -1.79 | +11.31 | +1.88 |
| OC-OC | dispersion | -4.44 | -1.01 | +8.81 | +2.28 | -13.25 | -2.26 |
| CC-CC | vol proxy | +14.84 | +2.45 | -0.25 | -0.06 | +15.09 | +2.03 |

The paper's claim is that dispersion is the dominant, asset-class-general
conditioner and that VIX-like variables only work for equity index futures.
Here the ordering is reversed: the volatility proxy separates the reversal
strategies and dispersion does not. That is a *proxy*, not the VIX, and this
sample is four instruments over four years - but it points the opposite way from
the paper's headline, and it points there consistently.

One honest caveat on dispersion itself. With N = 4 a cross-sectional standard
deviation has three degrees of freedom, and it leans on the two volatile legs
whether or not the returns are standardised first (correlation with gold's own
absolute overnight move: 0.47 raw, 0.53 standardised). This is the place where
the narrow universe genuinely binds, and the dispersion result is the one
finding here that a wider panel could plausibly overturn.

## 6. The placebo eats CO-OC and not CC-CC

Same construction, signals permuted **across instruments** within each day. Same
marginal distributions, same weighting, same turnover, same cost, no
cross-sectional information. Ten seeds:

| variant | real net | t | placebo mean | placebo sd | seeds beating real |
| --- | ---: | ---: | ---: | ---: | ---: |
| CC-CC | +6.60 | +1.76 | -0.91 | 3.65 | 0 / 10 |
| OO-OO | +0.56 | +0.16 | -2.79 | 3.11 | 1 / 10 |
| OC-OC | +3.26 | +1.12 | -1.94 | 3.41 | 0 / 10 |
| **CO-OC** | **-1.05** | **-0.34** | **-1.26** | **1.70** | **5 / 10** |

Five of ten. CO-OC is its own null. Whatever CC-CC and OC-OC have is at least
cross-sectional in origin - the shuffle destroys it - which is what makes
section 7 worth reading rather than section 6 being the end of the study.

The paper's investor-heterogeneity falsification check (their Table 6) behaves
as the paper expects, which is to say it finds nothing usable. AB_NR and AB_PR
at 10-, 20- and 30-day intervals produce t-statistics of -0.07, -1.29, -2.60 and
+0.74, +1.70, +0.72 respectively. The -2.60 is on **19 rebalances**; it is not a
result, it is a small number pretending to be one, and it is reported here only
so that nobody rediscovers it and gets excited.

## 7. What CC-CC actually is: 2020-2021

The strongest cell in the study, split at the sample midpoint:

| variant | 2020-01 .. 2022-01 | t | 2022-01 .. 2023-12 | t |
| --- | ---: | ---: | ---: | ---: |
| CC-CC | +13.90 | +2.52 | -0.72 | -0.14 |
| OO-OO | +2.00 | +0.41 | -0.87 | -0.17 |
| OC-OC | +9.51 | +2.36 | -3.01 | -0.72 |
| CO-OC | +6.44 | +1.65 | -8.56 | -1.89 |

The whole family lives in the first half and dies in the second. The first half
is the COVID crash and the reflation that followed - the highest cross-asset
volatility in the corpus, and the regime in which any mean-reversion strategy
looks like a genius. This is the same finding the session-breakout and level-
interaction studies in this directory reached by different routes, and it is
becoming the corpus's most reliable regularity: **cross-asset reversal on this
universe is a 2020-2021 phenomenon.**

## 8. Multiplicity, and what is left

Four variants are four hypotheses. Corrected together (rank, net, dev):

| variant | p | Bonferroni | Benjamini-Hochberg |
| --- | ---: | ---: | ---: |
| CC-CC | 0.0780 | 0.3121 | 0.3121 |
| OC-OC | 0.2628 | 1.0000 | 0.5256 |
| OO-OO | 0.8720 | 1.0000 | 0.8720 |
| CO-OC | 0.7332 | 1.0000 | 0.8720 |

Nothing clears anything. The best raw p-value in the study is 0.078 before
correction, on a strategy that section 7 shows is a single regime.

The validation split (366 sessions) was read once, and agrees without rescuing
anything: CC-CC nets +6.37 bps (t = +1.17, Sharpe 0.90), CO-OC nets +2.09 bps
(t = +0.57, Sharpe 0.46) - CO-OC's sign flips positive, which is what a
coefficient of zero does across samples. Every corrected p-value on validation
exceeds 0.4.

**The test split was not read.** Nothing here earned it.

## 9. The one place the paper might still be right

Weekly frequency is the only cut where CO-OC leads its family. 204 weeks,
Monday's open to Friday's close for the intraday leg, previous Friday's close to
Monday's open for the overnight leg:

| weighting | variant | gross | t | net | t | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rank | CO-OC | +19.08 | +1.19 | +17.05 | +1.07 | +0.54 |
| rank | OC-OC | +16.06 | +1.01 | +14.00 | +0.88 | +0.43 |
| rank | CC-CC | -2.41 | -0.16 | -3.65 | -0.24 | -0.10 |
| demean | OC-OC | +28.07 | +1.59 | +25.97 | +1.47 | +0.70 |

t = +1.07 on 204 observations is not a finding. But it is the one place the
paper's ordering survives, the cost drag is a fifth of the daily version's, and
the weekend is the only genuinely closed window these instruments have - which
is the closest this corpus comes to the illiquid-overnight mechanism the paper
is actually describing. If any part of this paper is worth another look on a
wider universe, it is this row.

## 10. What would change the answer

1. **A real cross-section.** Ten or more instruments per asset class is the
   experiment the paper ran and this one did not. The dispersion result in
   particular is a four-asset artifact waiting to be overturned; the return
   result is not, because a placebo that beats the strategy half the time does
   not become significant with more assets.
2. **Genuine futures with a genuine settlement.** A CFD's overnight is a
   convention; a contract's is a fact, with a settlement price, a pit close and
   real illiquidity in between. The mechanism the paper proposes needs that
   illiquidity to exist.
3. **The VIX, not a proxy.** Section 5's most surprising result - that a
   volatility conditioner separates these strategies and dispersion does not -
   rests on realised volatility standing in for implied. That substitution is
   defensible and it is not the same variable.
4. **A sample that is not half 2020.** Section 7 is the study's real finding and
   it is a statement about the corpus, not about the paper.

## 11. Reproducing

```bash
python scripts/backtests/backtest_overnight_reversal.py                    # every section, dev then validation
python scripts/backtests/backtest_overnight_reversal.py --only variants    # the core table
python scripts/backtests/backtest_overnight_reversal.py --only dispersion  # the mechanism regressions
python scripts/backtests/backtest_overnight_reversal.py --only placebo     # the shuffle, and AB_NR
python scripts/backtests/backtest_overnight_reversal.py --only weekly subperiods
python -m pytest tests/test_overnight_reversal.py                # timing, sign and cost asymmetry
```

The whole study runs in about a minute. The test suite carries a **positive
control** - a synthetic panel with a planted overnight-to-intraday reversal,
which CO-OC must find and the other variants must not - so the negative result
above is a statement about the data rather than about the implementation.

Inference throughout is `src/qlab/stats.py`: Newey-West standard errors on every
mean, Lo (2002)'s GMM standard error on every Sharpe, and the
Politis-Romano stationary bootstrap wherever a statistic has no closed form.
