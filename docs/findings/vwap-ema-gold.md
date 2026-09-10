# The regime-filtered VWAP/EMA gold strategy - evaluation

Subject: Bhatti (2026), *A Regime-Filtered Intraday Trading Framework for Gold:
Integrating VWAP Microstructure and EMA-Based Dynamic Exit Mechanisms*
(SSRN 6650958). Six machine-testable conditions on XAU/USD 15-minute candles:
price on the right side of a 200 EMA and of the session VWAP, a pullback that
touches the 50 EMA and closes back off it, a pin-bar or engulfing rejection
candle, above-average volume, and an above-average range. Stop at
`L_signal - 0.5 x ATR14`; target 3R; between them a trailing stop that fires
**only on a candle close** beyond the 50 EMA, tightening to the 20 EMA past
2.5R. Reported: **+0.414R expectancy, 45.3% win rate, profit factor 1.76,
annualised Sharpe 3.99, 5.1% maximum drawdown** over 247 trades.

Method: implemented in `src/qlab/strategies/vwap_ema.py`, resolved against the
**tick tape** rather than bars, on XAUUSD with USTEC and two FX majors as
breadth checks. **486 signals on dev (2020-2023), 170 on validation
(2024 - H1 2025).** Entries and exits cross a real bid and a real ask;
commission comes from the contract terms; slippage from the measured cost model.

**Verdict: rejected. The rule set has no measurable edge on gold, before or
after costs. Dev returns -0.035R per trade (t = -0.63), validation +0.067R
(t = +0.68); pooled over 656 trades it is -0.008R (t = -0.17, bootstrap 95% CI
[-0.097, +0.084]). The cost-free edge at the mid is +0.032R, so this is not a
strategy that costs killed - there was nothing there to kill. No cell of a
144-cell exit sweep survives its own multiplicity correction. The held-out test
split was not spent.**

The paper's headline numbers are not a measurement, and Section 6.1 says so.

---

## 0. There is no backtest in this paper

This has to come first, because it determines what the rest of the document can
possibly be. The paper's Section 6.1 describes its 247 trades as *"calibrated to
XAU/USD 15-minute data characteristics"*, with outcomes *"parameterised from the
strategy's structural logic"*: full wins with probability 0.30, partial wins
0.20, breakevens 0.08, full losses 0.42.

Those four probabilities are inputs. Everything in Tables 3 and 4 - the 45.3%
win rate, the +0.414R expectancy, the 1.76 profit factor, the 3.99 Sharpe, the
5.1% drawdown - follows arithmetically from them plus the 1:3 payoff and the
assumed cost. **No gold price enters the calculation.** The equity curve in
Figure 1 is a Monte Carlo draw, and the "benchmark comparison" in Section 6.5 is
that draw against another draw.

So there is nothing here to replicate, and this document does not claim to have
failed to replicate anything. What the paper *does* provide, and provides
unusually well, is a completely specified rule set: Section 3.4 gives six
conditions as inequalities, Eq. 4 gives the stop, Section 4.2 gives the trail.
That is enough to run it. This report is the measurement the paper does not
contain.

## 1. What this corpus can and cannot say

**Volume is quote updates, not contracts.** The Exness feed quotes bid and ask
with no size, so C5's volume test and the VWAP weights use `n_ticks` - how many
quote revisions the broker published in the bar. This is not a workaround
grafted onto a stock strategy: it is exactly what MT5 reports as "volume" on a
spot gold chart, so an implementation on the paper's own stated platform would
read the same quantity. It is still a proxy and is labelled as one throughout.

**The news filter is absent.** The paper excludes entries within 15 minutes of
NFP, CPI, FOMC and GDP releases. This corpus holds no macro calendar, so the
filter is not implemented, which makes every number here slightly *pessimistic*
relative to the paper's intent. Section 7 below is the substitute.

**The 0.24R cost assumption is roughly ten times too large.** The paper's
Table 2 puts the round turn at 0.24R ($2.40 against a $10 stop). Measured on
this account, 1R has a median of $5.23 on dev and $7.95 on validation, and the
round turn is $20 and $17 per standard lot - **0.038R and 0.022R**. The paper
is conservative on costs, which is worth saying plainly: this rejection is not a
cost argument.

## 2. Why this is resolved on ticks

1R is half a 14-period ATR plus the distance from the fill to the signal
candle's extreme - a median of $5.23 on dev, against 15-minute gold candles
whose own range routinely exceeds that. A bar-level engine holding both the
entry and the stop inside one candle would have to *assume* which came first,
and on this geometry that assumption is worth more than the edge being measured.

Entry, stop and target are therefore resolved against the tick tape. Only the
trail is a bar-close event, because the paper defines it as one. Three sanity
checks fall out of the implementation and confirm it: stop exits average
**-1.029R**, target exits **+2.981R**, and the shortfall in each case is exactly
the commission.

A choice worth stating: a long's target is a sell limit, so it fills when the
**bid** reaches it, not the ask. That is one spread more conservative than the
convention used elsewhere in this project and is pinned by test.

## 3. The rule exactly as written

| | dev (2020-2023) | validation (2024-H1 2025) | pooled |
| --- | --- | --- | --- |
| trades | 486 | 170 | 656 |
| mean net R | **-0.035** | **+0.067** | **-0.008** |
| Newey-West t | -0.63 | +0.68 | -0.17 |
| bootstrap 95% CI | [-0.137, +0.068] | [-0.106, +0.246] | [-0.097, +0.084] |
| win rate | 30.5% | 38.2% | 32.5% |
| profit factor | 0.93 | 1.16 | - |
| edge at the mid | +0.010R | +0.093R | +0.032R |

The two splits disagree in sign and neither is distinguishable from zero. The
pooled interval is tight enough to be informative: it excludes anything larger
than +0.084R per trade, which is a fifth of the paper's claim.

Against the paper's own Table 3, on dev:

| metric | paper | measured |
| --- | --- | --- |
| trades | 247 (one year) | 486 (3.9 years) |
| win rate | 45.3% | 30.5% |
| expectancy | +0.414R | -0.035R |
| average win | +2.12R | +1.47R |
| average loss | -1.07R | -0.69R |
| profit factor | 1.76 | 0.93 |
| Sharpe | 3.99 | -0.27 |
| max drawdown | 5.1% | 30.8% |
| total return | +102.2% | -19.5% |

The signal frequency is the one thing that lines up in order of magnitude: 124
signals a year on dev and 114 on validation, against the paper's 247. Everything
else diverges, and the win rate diverges in the direction that matters - the
paper's assumed 0.30 probability of reaching 3R is measured at **0.093**.

Against Table 4:

| outcome | paper n | paper % | measured n | measured % | measured R |
| --- | --- | --- | --- | --- | --- |
| full win (3R) | 70 | 28.3% | 45 | 9.3% | +2.98 |
| partial win | 42 | 17.0% | 84 | 17.3% | +0.97 |
| breakeven | 24 | 9.7% | 47 | 9.7% | -0.02 |
| loss | 111 | 44.9% | 310 | 63.8% | -0.75 |

The middle two rows match almost exactly. The outer two do not, and they are the
two that carry the expectancy: three times too few full wins, and half again too
many losses.

## 4. The trail is not the problem, and not the solution either

The paper's stated mechanical contribution is the close-conditioned trail: gold
wicks through the 50 EMA transiently, so an exit conditioned on closes filters
false stops that a fixed-distance stop would take. The implementation honours
this exactly - a test drives the tape well below the EMA mid-bar and asserts the
position survives.

It does what the paper says, and it does not rescue the strategy. On dev the
trail ends 37.9% of trades at a mean of **-0.51R**, against stop-outs at -1.03R:
it is genuinely cutting losses that would otherwise have been full ones. But
turning it off makes the results *worse*, not better, across almost the whole
exit grid. The mechanism works; there is simply no edge for it to protect.

## 5. The exit surface: flat, everywhere

Six stop widths x six targets x trail on/off x session-flat on/off - 144 cells,
each on the same shared entries so no cell is handed a different fill from its
neighbour. Dev, mean net R:

```
  --- trail=on, flat=on (the paper's row of variants) ---
                1R     1.5R       2R       3R       5R     hold
   0.25     -0.079   -0.065   -0.100   -0.043   -0.007   +0.021
   0.50     -0.056   -0.087   -0.083   -0.035*  -0.000   -0.000
   0.75     -0.055   -0.066   -0.042   -0.001   +0.017   +0.034
   1.00     -0.058   -0.045   -0.017   +0.005   +0.020   +0.021
   1.50     -0.036   -0.006   +0.010   +0.034   +0.034   +0.027
   2.00     -0.010   +0.019   +0.029   +0.027   +0.040   +0.032
                                    * the paper's cell
```

The whole surface lies between -0.10R and +0.05R. The best cell on dev is
+0.048R at t = +1.27; a single cell would need **t = 3.57** to survive searching
144 of them. On validation the best cell is +0.111R at t = +1.55 against the
same threshold, and it is a *different* cell - stop 0.5 / target 1R rather than
stop 2.0 / target 5R. Two independent searches over the same grid picking
opposite corners is what a flat surface looks like.

This is the cleanest form of the verdict. When a paper fixes one cell in
advance, the interesting question is where that cell sits: on a broad plateau
the geometry is real, at an isolated peak it is fitted. Here it sits on a
plateau at zero.

## 6. Do the six conditions do any work?

The attrition table shows what each condition removes. On dev, of 25,008
tradeable session bars, the long chain runs 45.9% -> 32.2% -> 4.9% -> 2.3% ->
1.14% -> 1.03%. C3 - the 50 EMA pullback - is by far the most selective; C1, C2
and C6 each discard roughly half of what reaches them.

Selectivity is not edge:

| variant (dev) | n | mean R | t |
| --- | --- | --- | --- |
| C4 as written | 486 | -0.035 | -0.63 |
| C4 = pin bar only | 102 | -0.100 | -0.96 |
| C4 = engulfing only | 420 | -0.026 | -0.44 |
| **C4 removed entirely** | **1030** | **-0.029** | **-0.60** |
| longs only | 258 | -0.011 | -0.14 |
| shorts only | 228 | -0.062 | -0.72 |

Dropping the rejection-candle condition doubles the trade count and changes the
mean by 0.006R. Whatever C4 is selecting, it is not selecting on outcome.

**The placebo** keeps the session window, the trade count, the long/short mix
and the entire exit geometry, and randomises only *when* the entry happens.
Thirty draws:

| split | placebo mean | placebo sd | the rule | percentile |
| --- | --- | --- | --- | --- |
| dev | -0.070R | 0.062 | -0.035R | 70th |
| validation | -0.081R | 0.091 | +0.067R | 90th |

The rule beats a random entry in both splits, which is the one mildly
encouraging result in this document - and it does so by less than one placebo
standard deviation on dev. Note also that the placebo mean is *negative* in both
splits: entering gold at a random moment in the US session and managing the trade
with this exit geometry loses about 0.07R. Most of what the six conditions
achieve is getting back to zero.

## 7. Hours, the anchor, and breadth

**Hours.** The news filter cannot be implemented here, so the substitute asks
whether the release hours drag. On dev the 18:00 UTC bucket - which holds FOMC -
is the worst at -0.283R (t = -1.79, n = 23), and 13:00 and 14:00, which hold NFP
and CPI, sit at -0.025R and -0.046R on 388 trades between them. Excluding news
might have removed a small negative, on a strategy whose total is a small
negative. It is not the missing ingredient.

**The anchor.** The paper writes "the New York session open (13:30 UTC)". Those
are two different anchors for the five months the US is not on daylight saving
time. Both were run, with the session window moved onto whichever clock the
anchor names:

| | dev | validation |
| --- | --- | --- |
| 13:30 UTC fixed (as written) | -0.035R (n=486) | +0.067R (n=170) |
| 09:30 New York (as meant) | -0.107R (n=47) | +0.081R (n=143) |

Neither reading rescues it. The dev row for the New York anchor is on 47 trades
and should not be read as anything but "not obviously different".

**Breadth.** The same six conditions on three instruments the paper never
claims:

| | dev | validation |
| --- | --- | --- |
| XAUUSD (the claim) | -0.035R (n=486) | +0.067R (n=170) |
| USTEC | -0.053R (n=809) | -0.004R (n=338) |
| EURUSD | -0.073R (n=359) | +0.156R (n=127) |
| USDJPY | -0.080R (n=302) | -0.072R (n=162) |

Eight instrument-splits, seven of them within noise of zero, three of them
positive. There is no instrument on which these rules work and gold is not
distinguished among them - which matters, because the paper's entire theoretical
case (Sections 2.1-2.3) is about properties specific to gold.

## 8. The VWAP milestone protocol is internally contradictory

Section 4.3 manages the trade as price *approaches* VWAP after entry, cutting
50% on a "compressed approach" and holding through an "impulsive" one. But C2
requires price to already be beyond VWAP at entry. For a long, VWAP is therefore
*below* the fill, and price returning to touch it is an adverse move, not a
milestone on the way to target.

Measured rather than assumed: the post-entry VWAP touch fires on **385 of 486
dev trades (79%)** and 122 of 170 on validation, at a mean floating **-0.44R**
when it does. The protocol is not unreachable - it is reached constantly, and
always from the wrong side. It is recorded per trade as `vwap_touch_r` and kept
out of the trade path, because there is no reading of Section 4.3 that is both
faithful and coherent.

## 9. What this does not say

- **It does not say the paper's Sharpe of 3.99 is wrong.** It is correct
  arithmetic on its stated assumptions. It says those assumptions are not
  gold's.
- **It does not test the news filter**, which is a real component and is absent
  here. If it is the load-bearing part, this study cannot see it.
- **It is one instrument over 5.4 years**, of which the paper's own sample year
  (2024) is inside the validation split and contributes 170 trades on its own.
- **It does not rule out a +0.05R edge.** The pooled interval reaches +0.084R.
  What it rules out is anything near the claim.
- **The volume proxy is a proxy.** C5 uses tick counts. If real gold futures
  volume separates these bars differently, C5 is being tested in a weakened form.

## 10. If you want to keep pulling this thread

The one asymmetry worth a look is that the placebo mean is negative in both
splits while the rule is not: entering at a random moment in the US gold session
and managing with this geometry loses money, and the six conditions recover it.
That is a filter with real information in it and no edge on top - consistent with
the conditions selecting bars that are merely *less bad*, e.g. by avoiding the
worst of the spread or the thinnest minutes. If so, the interesting object is not
this strategy but whatever makes a random US-session gold entry lose 0.07R.

The longs-only cut on validation (+0.231R, t = +1.66, n = 88) is the only cell in
this whole study that comes close to significance. It is one of roughly forty
comparisons here and gold rose steadily through the validation window, so it is
noted and not pursued.

## 11. Reproducing

```bash
python scripts/backtests/backtest_vwap_ema.py --splits dev validation --placebo-draws 30
```

Writes `reports/strategies/vwap_ema.json` and per-trade tapes to
`reports/strategies/vwap_ema_trades/`. Roughly four minutes; the tick sweep
dominates. Tests: `pytest tests/test_vwap_ema.py` (46 cases, pinning the six
conditions, the side of the book each order touches, the precedence of the three
exit clocks, and the close-only trail).

The `test` split was **not** read.
