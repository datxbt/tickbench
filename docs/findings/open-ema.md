# The New York open EMA rule - evaluation

Subject: a rule circulated by its creator for NAS100 on a 5-minute chart. Wait
for the 09:30 New York open; take the first 5-minute candle, 09:30 -> 09:35; if
it closes **above** the 12-period EMA go long, **below** go short; trail the
stop behind the position and "stay in as long as momentum continues"; risk 1%
per trade. Reported: 5-minute NAS100 data 2019-2026, **1,448 trades, +982%
total return, 57% win rate, 1.29 profit factor, 19.7% maximum drawdown**. The
creator's framing is "probability, not prediction" - a statistical edge over
many trades rather than a claim about any single day.

Method: implemented in `src/qlab/strategies/open_ema.py`, run by
`scripts/backtests/backtest_open_ema.py`, resolved against the **tick tape**
rather than bars, on USTEC (the Exness NAS100 CFD). **960 sessions on dev
(2020-02 to 2023-12), 368 on validation (2024-01 to 2025-06), 1,328 pooled**
against the creator's 1,448. Fills cross a real bid and a real ask; commission
is the contract term; slippage comes from the measured cost model. Because the
rule names an entry and a side but no stop distance, the exit is **swept**: three
families at seven distances, twenty-one cells, every one of them priced on the
same fills.

**Verdict: the rule does not replicate, and the reported statistics are not
reachable by the exit it describes.** As specified - trailing stop - it earns
**+0.048 R per trade pooled** (t = +1.65, bootstrap p = 0.084, q = 0.19 across
the grid), which is not distinguishable from a coin flip taking the same fills
at the same minute (difference +0.039 R, p = 0.286). At 1% risk that compounds
to **+74%**, not +982%. The reported 57% win rate is not attainable anywhere in
the sweep: the highest is 52.3%, and the reported pair (57% wins, PF 1.29)
implies winners the same size as losers, which is the opposite of the payoff
profile a trailing stop produces. The one cell that does reproduce the headline
return - **+942% pooled** - uses a **fixed** stop with no trail, which is the
opposite of the stated rule, and even that cannot beat the coin control
(p = 0.170). **The locked test split was not spent.**

Two things do hold up, and they are worth separating from the verdict:

| holds | does not hold |
| --- | --- |
| the side has the right sign in both splits (+0.023, +0.042 ATR to the bell) | neither is significant (t = +1.20, +1.24) |
| the edge orders correctly in the signal's own strength | the coin control orders the same way |
| every calendar year is positive in the best cell | so is the always-long control's, and the coin's |

---

## 0. The rule collapses before it is tested

The comparison is "the candle closes above the 12-period EMA". With the usual
recursion `e_t = a*c_t + (1-a)*e_{t-1}`:

```
c_t > e_t
c_t > a*c_t + (1-a)*e_{t-1}
(1-a)*c_t > (1-a)*e_{t-1}
c_t > e_{t-1}                    for any a < 1
```

**"Closes above its own EMA" is the same event as "closes above the EMA as it
stood before the candle opened", for every span.** Two consequences.

The first is procedural, and it is good news: the one genuine ambiguity in the
specification - whether the signal candle is included in its own average -
**dissolves**. Both readings give the identical signal on every bar, so there is
nothing to choose and nothing to defend. `signal_equivalence` and
`tests/test_open_ema.py` pin it on random walks, strict trends, flat series and
alternating series.

The second is substantive: **the 12 does no work in choosing the side.** The
span survives only in *how far* the close sits from the level, never in which
side the rule takes. So this is not an EMA crossover strategy. It is a momentum
print with one bit of output: did the first five minutes of the cash session
close above where the previous hour had settled?

That reframing is what the rest of the study tests, and it sets the prior. A
one-bit read on a five-minute candle, turned into a day-long directional bet, is
close to a coin flip by construction - so the controls are not a formality here,
they are the experiment.

## 1. The signal, before any trading

| | dev | validation |
| --- | --- | --- |
| sessions | 960 | 368 |
| long | 51.1% | 50.0% |
| doji (close exactly on the EMA) | 0 | 0 |
| median \|close - EMA\| | 0.083 ATR | 0.073 ATR |
| mean 14-day ATR | 258 pts | 332 pts |

Two things to take from this. **The rule is not a disguised long.** At 51.1% and
50.0% long it is very nearly balanced, which rules out the first explanation
anyone should reach for on an instrument that roughly tripled over the sample -
that the reported return is NAS100 drift wearing a signal. It is not.

And **the signal is marginal by construction.** The median close sits 0.083 ATR
from the EMA - about 23 index points on a 278-point ATR. Half of all decisions
are made on a gap smaller than that. This is the one-bit read of section 0,
taken at its thinnest.

## 2. The raw signal, before any exit

Forward mid returns in ATR units, so the geometry of the exit and the cost of
the round turn are held apart from the question of whether the side was right.

| rule | fwd 15m | 30m | 1h | 2h | to the bell | t(bell) |
| --- | --- | --- | --- | --- | --- | --- |
| **dev** | | | | | | |
| ema | -0.0034 | +0.0109 | +0.0182 | +0.0159 | **+0.0231** | +1.20 |
| coin | -0.0028 | -0.0052 | -0.0090 | +0.0014 | -0.0006 | -0.03 |
| long | +0.0032 | +0.0037 | +0.0043 | +0.0021 | +0.0184 | +1.05 |
| **validation** | | | | | | |
| ema | +0.0012 | +0.0114 | +0.0373 | +0.0325 | **+0.0418** | +1.24 |
| coin | +0.0160 | +0.0222 | +0.0339 | +0.0279 | +0.0582 | +1.70 |
| long | -0.0060 | -0.0069 | -0.0000 | +0.0004 | -0.0049 | -0.17 |

The rule's side is positively signed in both splits and grows with horizon,
which is the shape the claim predicts. Neither split is significant, and on
validation the **coin flip beats it** (+0.058 against +0.042). Mean MFE is
+0.454 ATR against mean MAE -0.438 ATR on dev - a near-symmetric excursion
envelope, which is the first hint that no exit rule is going to find much
asymmetry to harvest here.

## 3. The exit surface: the creator's exit is the worst of the three

Twenty-one cells, pooled dev + validation, net of cost, R per trade. `q` is the
Benjamini-Hochberg correction across the whole grid, because the largest `t` in
a table of twenty-one is a maximum over twenty-one draws and its nominal p-value
is not the p-value of any decision a reader would make.

| style | dist (ATR) | mean R | t | q | win% | PF | payoff | stop% | hold (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trail | 0.05 | -0.0505 | -1.54 | 0.200 | 35.5 | 0.89 | 1.62 | 100.0 | 2 |
| trail | 0.10 | -0.0160 | -0.56 | 0.577 | 37.0 | 0.96 | 1.64 | 100.0 | 8 |
| **trail** | **0.15** | **+0.0517** | **+1.71** | 0.189 | 37.0 | 1.13 | 1.93 | 99.6 | 23 |
| trail | 0.25 | +0.0475 | +1.65 | 0.189 | 37.0 | 1.13 | 1.92 | 93.8 | 78 |
| trail | 0.50 | +0.0388 | +1.68 | 0.189 | 43.0 | 1.11 | 1.48 | 62.2 | 230 |
| trail | 0.75 | +0.0315 | +1.69 | 0.189 | 48.9 | 1.12 | 1.17 | 32.8 | 318 |
| trail | 1.00 | +0.0210 | +1.43 | 0.215 | 50.9 | 1.10 | 1.06 | 16.2 | 357 |
| fixed | 0.05 | +0.1963 | +1.69 | 0.189 | 9.6 | 1.20 | 11.23 | 90.3 | 52 |
| **fixed** | **0.10** | **+0.2238** | **+2.78** | **0.114** | 18.4 | 1.26 | 5.60 | 81.3 | 99 |
| fixed | 0.15 | +0.1203 | +1.92 | 0.189 | 24.2 | 1.15 | 3.61 | 75.0 | 133 |
| fixed | 0.25 | +0.0554 | +1.28 | 0.256 | 34.8 | 1.09 | 2.04 | 61.5 | 198 |
| fixed | 0.50 | +0.0437 | +1.57 | 0.200 | 47.3 | 1.11 | 1.23 | 33.6 | 304 |
| fixed | 0.75 | +0.0370 | +1.83 | 0.189 | 51.3 | 1.13 | 1.07 | 16.4 | 351 |
| fixed | 1.00 | +0.0204 | +1.26 | 0.256 | 51.8 | 1.09 | 1.01 | 8.5 | 371 |
| be_trail | 0.05 | -0.0645 | -1.76 | 0.189 | 46.0 | 0.89 | 1.04 | 100.0 | 2 |
| be_trail | 0.10 | +0.0264 | +0.77 | 0.461 | 50.2 | 1.05 | 1.04 | 100.0 | 13 |
| be_trail | 0.15 | +0.0298 | +0.87 | 0.424 | 48.3 | 1.06 | 1.13 | 99.3 | 35 |
| be_trail | 0.25 | +0.0588 | +1.87 | 0.189 | 50.1 | 1.12 | 1.12 | 88.3 | 119 |
| be_trail | 0.50 | +0.0402 | +1.50 | 0.201 | 50.5 | 1.10 | 1.08 | 45.6 | 283 |
| be_trail | 0.75 | +0.0347 | +1.77 | 0.189 | 52.3 | 1.12 | 1.02 | 19.9 | 346 |
| be_trail | 1.00 | +0.0192 | +1.21 | 0.265 | 52.0 | 1.09 | 1.00 | 9.2 | 370 |

`trail` ratchets from the first tick; `fixed` never moves and the bell closes
what survives; `be_trail` holds the initial stop until the trade is one distance
in favour and trails from there. All three risk exactly one distance at the first
tick, so `R` means the same thing in every row (`tests/test_open_ema.py`).

**Not one cell clears q = 0.10.** The best in the table is q = 0.114.

**The stated exit is the weak one.** Trailing tops out at +0.052 R per trade;
not trailing at all tops out at +0.224 R, four times as much. A tight trail is
outright negative - `trail` at 0.05 ATR loses 0.051 R per trade pooled, and on
validation alone it loses 0.146 R at t = -2.46. The stop-out rate says why: at
0.05 and 0.10 ATR the trail is hit on **100%** of sessions, with a mean hold of
2 and 8 minutes. "Stay in as long as momentum continues" describes an exit that,
at any distance tight enough to matter, does not stay in at all.

### Cost is the whole story at the tight end

The same grid at the mid tells a different story from the same grid net:

| cell | at the mid (dev) | net of cost (dev) | cost |
| --- | --- | --- | --- |
| trail @ 0.05 | +0.1517 R (t +3.93) | **-0.0139 R** (t -0.36) | 0.158 R |
| trail @ 0.10 | +0.0759 R (t +2.17) | -0.0055 R | 0.078 R |
| fixed @ 0.05 | +0.4209 R (t +2.95) | +0.2580 R | 0.158 R |

The round turn is about **1.9 index points** - a 1.19-point spread, 0.63 points
of commission, 0.08 of slippage - and it barely changes with the stop distance,
so as a fraction of risk it is 0.158 R at 0.05 ATR and 0.007 R at 1.00 ATR. The
tightest trail has the highest gross edge in the entire study and loses money
anyway. Anyone backtesting this rule on 5-minute bars without a cost model would
find its most attractive cell to be its most negative one.

## 4. Controls: the rule as specified

`trail @ 0.25 ATR` - the creator's own exit, at the trailing distance that did
best on dev. Every rule below takes the same fill at the same instant; only the
side differs.

| | dev | validation | pooled |
| --- | --- | --- | --- |
| ema | **+0.0536** (t +1.60) | **+0.0317** (t +0.56) | **+0.0475** (t +1.65) |
| contra | -0.0640 | -0.0290 | -0.0543 |
| coin | -0.0218 | **+0.0890** | +0.0089 |
| long | +0.0148 | -0.0238 | +0.0041 |
| short | -0.0252 | +0.0265 | -0.0109 |
| ema - coin | +0.0754 (p 0.110) | -0.0573 (p 0.366) | +0.0386 (p 0.286) |
| ema - long | +0.0388 (p 0.284) | +0.0555 (p 0.436) | +0.0434 (p 0.162) |
| total at 1% risk | +58.1% | +10.1% | **+74.2%** |
| max drawdown | -19.3% | -25.8% | -34.9% |

Pooled bootstrap on the rule: +0.0475 R, 95% CI [-0.008, +0.108], p = 0.084.

The rule beats its inverse in both splits, which is the least the claim can ask
for. It does not separate from a coin flip in either, and on validation the coin
flip beats it outright. Pooled, the difference from the coin is +0.039 R with a
CI straddling zero.

The drawdown column is the one place the study lands near the reported figure -
-19.3% on dev against the reported -19.7% - but with +58% of return instead of
+982%, which is to say the reported risk was matched only by a version of the
strategy that made almost none of the reported money.

## 5. The cell that does reproduce the headline - and is not the rule

Ranking the dev surface by mean R picks `fixed @ 0.05`; ranking it by `t` picks
`fixed @ 0.10`. Both are carried forward. The second is the interesting one.

`fixed @ 0.10 ATR` - a fixed 28-point stop, no target, hold to the bell:

| | dev | validation | pooled |
| --- | --- | --- | --- |
| ema | **+0.2246** (t +2.45) | **+0.2217** (t +1.35) | **+0.2238** (t +2.78) |
| coin | +0.0398 | **+0.2540** | +0.0991 |
| long | +0.1057 | +0.0470 | +0.0894 |
| ema - coin | +0.1848 (p 0.110) | -0.0324 (**p 0.794**) | +0.1246 (p 0.170) |
| ema - long | +0.1188 (p 0.260) | +0.1747 (p 0.256) | +0.1343 (p 0.114) |
| win rate | 18.2% | 18.8% | 18.4% |
| profit factor | 1.26 | 1.26 | **1.26** |
| total at 1% risk | +456.9% | +87.1% | **+942.2%** |
| max drawdown | -37.9% | -37.7% | -37.9% |

Pooled bootstrap: +0.2238 R, 95% CI [+0.072, +0.374], p < 0.001. Positive in
every calendar year: +0.082 (2020), +0.238 (2021), +0.405 (2022), +0.152 (2023),
+0.182 (2024), +0.305 (2025).

So a cell exists that compounds to **+942%** at 1% risk against the reported
+982%, at a profit factor of **1.26** against the reported 1.29. Read alone, that
looks like a replication. Three reasons it is not.

**It is not the rule.** A fixed stop that never moves is the opposite of the
stated exit. The described trailing version of the same idea makes +74%.

**It does not beat the null.** The coin flip taking the same fills returns
+0.099 R pooled, and the difference is p = 0.170. On validation the coin
returned **+0.254 R against the rule's +0.222** - it won. Always-long returns
+0.089 R, p = 0.114 against the rule.

**It is a maximum over twenty-one cells.** Its q across the grid is 0.114.

The pooled compounded returns make a point worth keeping: the rule's +942% and
the coin's +112% come from mean per-trade edges of +0.224 R and +0.099 R that
are **not statistically distinguishable from each other**. Compounding at fixed
fractional risk turns a difference no test can resolve into an eight-fold
difference in the headline. A total-return figure is not evidence about a signal;
it is that signal's per-trade edge amplified by a sizing choice, and 1,328 trades
is more than enough to make the amplification enormous and nowhere near enough
to make the edge certain.

## 6. The reported win rate is not reachable by this exit

The reported pair is 57% wins at a profit factor of 1.29. Those two numbers
together imply an average win of `PF*(1-w)/w = 1.29*0.43/0.57 =` **0.97** average
losses - winners and losers about the same size.

That is not a shape a trailing stop makes. A trail cuts losers quickly and lets
winners run, which produces *fewer* wins with a payoff well above 1. The sweep
shows the trade-off directly: as the distance widens the win rate climbs from
35.5% to 50.9% while the payoff falls from 1.62 to 1.06, and the two never meet
where the report says they do.

- Highest win rate anywhere in the 21-cell sweep: **52.3%** (`be_trail @ 0.75`),
  and there the profit factor is 1.12 and the payoff 1.02.
- Cells reaching 57% wins and PF 1.29 together (within 2pp and 0.05): **0**.
- The cells that generate the reported *return* win 9.6% to 18.4% of the time,
  with payoffs of 5.6 to 11.2.

A 57%/1.29 pair is the signature of a **target-based** exit - something that
books a profit at a level rather than riding a trail. Whatever produced the
reported figures, it is not the exit the rule describes. This is the one finding
here that does not depend on the corpus, the instrument or the sample period: it
is an internal inconsistency in the reported statistics.

## 7. Does the signal's own strength do anything?

If the distance to the EMA carries information, the rule should do better where
that distance is larger. Asking for an *ordering* is much weaker than asking for
a significant mean, so it is a fair test of the mechanism. But a larger gap also
means a more volatile open, and a tight stop with no target has fatter tails on a
volatile day whichever way it is pointed - so the controls are sorted the same
way. Mean net R by quartile of |close - EMA|, pooled, 332 sessions per bucket:

| | q1 nearest | q2 | q3 | q4 furthest |
| --- | --- | --- | --- | --- |
| **fixed @ 0.10** | | | | |
| ema | +0.058 | +0.059 | +0.207 | **+0.572** |
| coin | +0.075 | +0.012 | -0.057 | **+0.366** |
| long | +0.065 | +0.003 | +0.128 | +0.162 |
| **trail @ 0.25** | | | | |
| ema | -0.004 | +0.039 | +0.064 | **+0.091** |
| coin | -0.022 | +0.021 | -0.029 | +0.066 |
| long | -0.074 | +0.042 | +0.100 | -0.051 |

The rule's ordering is monotone in both cells, which is what the mechanism
predicts. **But the coin's top quartile jumps too** - to +0.366 R in the fixed
cell - so most of the gradient is the volatility effect, not the signal. The
rule's margin over the coin is larger in q4 (+0.206 R) than anywhere else, which
is where a real effect would sit if there is one; with 332 sessions a bucket,
that is a hint and not a result.

## 8. The one specification choice that is genuinely open

The rule says "12-period EMA" on a 5-minute NAS100 chart without saying which
5-minute series. A chart is continuous, so the twelve bars behind 09:35 are the
hour from 08:40 - overnight and pre-market tape. That is the default above. The
alternative is a regular-session-only series, where the twelve bars behind 09:35
reach back into the previous afternoon.

| | continuous (default) | rth |
| --- | --- | --- |
| long share, dev | 51.1% | 55.9% |
| median \|close - EMA\|, dev | 0.083 ATR | 0.226 ATR |
| best dev cell, `fixed` family | +0.2246 R (t +2.45) | +0.0381 R (t +0.46) |
| best validation cell, `fixed` | +0.2217 R (t +1.35) | +0.0934 R (t +0.46) |

The `rth` surface sits on zero at every distance - dev runs from -0.050 to
+0.038 R. So what little the rule has **depends on the EMA including overnight
tape**, and the signal it is really taking is "did the cash open confirm or
reject the overnight drift", not "is price above its session average". Note the
direction of the effect: the `rth` variant has a signal nearly three times
larger by magnitude (0.226 ATR against 0.083) and no edge at all. A bigger
reading of the same indicator is not a better one.

## 9. What this does not say

- **It does not say 2019 and the last fourteen months behave like this.** The
  corpus starts 2020-01-29, so the creator's 2019 is unavailable, and
  2025-07-01 to 2026-09-01 is the locked test split, which was not read. The
  1,328 sessions here against their 1,448 are a different window as well as a
  shorter one.
- **It does not say their numbers are wrong on their data.** This is the Exness
  NAS100 CFD with its own spread, its own session boundaries and its own
  dividend-adjustment behaviour. A different NAS100 feed, or the futures, could
  differ - though not, per section 6, in a way that makes 57% wins compatible
  with a trailing stop.
- **It does not rule out an edge too small to see here.** The pooled CI on the
  rule as specified is [-0.008, +0.108] R per trade. An edge of +0.05 R is
  entirely consistent with these data; so is zero. Validation on its own could
  only have confirmed an edge of +0.483 R or larger at t = 2.0, which is four
  times the claimed expectancy - the claim's own +0.125 R would have registered
  at t = +0.52 there. The power was stated before the split was read.
- **It does not test the creator's actual algorithm.** "Fully automated with
  trailing stops" is a description, not a specification. Three families at seven
  distances is a serious attempt at covering it, and the sweep's best trailing
  cell is +0.052 R, but some particular trail this study did not try may do
  better.

## 10. If you want to keep pulling this thread

The interesting residue is not the EMA. It is that a **fixed 0.10-ATR stop, no
target, held to the bell** earns +0.224 R per trade pooled on USTEC, positive in
all six calendar years, whichever side it is pointed - the coin control makes
+0.099 R and always-long +0.089 R on the same geometry. That is a statement about
the shape of the USTEC cash session, not about a signal: a tight stop on a
fat-right-tailed index buys a cheap option on the day. It is also a statement
this study cannot make strongly, because the cell was selected from twenty-one
and q = 0.114.

Two routes that would not just re-read the same data:

1. **Pre-register the fixed-stop geometry as its own hypothesis** with no signal
   at all - direction from a coin, or always long - and size the power before
   looking. The question is whether a 0.10-ATR stop held to the bell has positive
   expectancy on an index, which is a cleaner question than this one and needs no
   EMA.
2. **Ask the creator what the exit actually is.** The 57%/1.29 pair points at a
   profit target, and a target changes the study. The exit surface here can be
   extended with R-multiple targets cheaply, since the fills are already
   resolved - `qlab.strategies.opening_range` already sweeps targets on the same
   09:35 clock.

The test split stays unspent. Nothing here is a survivor: the rule as specified
does not clear a coin flip, and the cell that reproduces the headline is not the
rule.

## 11. Reproducing

```bash
python scripts/backtests/backtest_open_ema.py            # the whole study
python scripts/backtests/backtest_open_ema.py --cached   # re-report from parquet
```

Roughly four minutes over the tick tape for both splits; trade frames land in
`reports/strategies/open_ema_USTEC_{dev,validation}.parquet`. Sections in the
output map one-to-one onto the sections above. The script has no flag that reads
the test split.
