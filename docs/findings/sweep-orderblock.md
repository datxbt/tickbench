# Liquidity sweeps and order blocks (UAlgo *Price Action Toolkit Lite*) - evaluation

Subject: two components of a TradingView indicator (Pine v6, © UAlgo,
CC BY-NC-SA 4.0), turned into trades and run tick by tick on EURUSD, USDJPY,
XAUUSD and USTEC at **twelve timeframes from 1m to D1**.

The indicator draws and does not trade, so every trading choice below was
made - and fixed - before any outcome was computed. Implementation:
`src/qlab/strategies/sweep_orderblock.py`. Driver:
`scripts/backtests/backtest_sweep_orderblock.py`. Tests:
`tests/test_sweep_orderblock.py`.

---

## Pre-registration (written 2026-09-12, before any P&L, win rate or forward return was computed)

The only numbers seen before this section was written are **order counts** from a
bars-only dry run (dev: 91,239 sweep orders, 419,925 block orders, across the
four instruments and twelve timeframes). No tick was resolved.

### The two setups

```text
SW  liquidity sweep
    Level:    ta.pivothigh/low(30, 30), confirmed 30 bars late; <= 7 live per side,
              an 8th pushes the oldest out untested.
    Event:    the first bar whose high (low) trades beyond a live level retires it;
              if that bar closes back inside the level, it is a sweep (the 'x').
    Trade:    against the sweep, market, at the standing quote at the bar's close.
    Stops:    s0 = the sweep bar's wick extreme; s1 = s0 + 0.25 ATR14.

OB  order-block retest
    Swings:   one-sided zigzag, length 9 (high[9] >= highest(high, 9)), trend flips
              on the opposite swing - exactly as the script.
    Break:    a close beyond the latest swing (BoS if the previous break went the
              same way, CHoCH if not).
    Block:    bearish = highest high of the bars after the swing low through the
              break bar; zone [max - ATR14, max]. Bullish mirrored.
    Live:     newest 2 per side shown; a shown block is deleted on a close through
              its far edge; hidden, deleted or 100 bars old cancels the order.
    Trade:    resting limit at the zone's near edge, in the block's direction.
    Stops:    s0 = the far edge (risk = 1 ATR); s1 = s0 + 0.5 ATR.

Both      targets 1R, 2R, 3R, or none ("hold"); life cap 60 bars after the fill.
          8 exits per setup, all resolved on the same fill.
```

Departures from the Pine source, all stated in the module docstring: the
bar-count lookback is done by index (the script's time-difference arithmetic is
wrong across gaps); a hidden block is not re-armed if it later reappears; the
block order has a 100-bar life; pivot ties are left-strict, right-weak.

Bars: up to 1h on the UTC grid; 2h, 4h and D1 cut at 17:00 New York.

Fills: bid/ask from the tape, published commission, `CostModel` slippage
(250 ms, 0.5 adverse) on every market leg, none on resting limits. Cost
profiles are measured on the split being evaluated. Ties between a stop and a
target on one tick go to the stop. Tick windows are clamped at the split's own
end. Swap is not in the quote feed: each trade records the 17:00 New York rolls
it crosses, and 4h/D1 results are reported a second time charged at the
measured overnight basis (`qlab.rollover.basis`).

### The null

`matched`: size-matched ordinary bars (a deterministic timestamp hash), same
geometry, direction by a hash coin. For SW, a random bar entered at market with
the stop at its own wick. For OB, a zone built by the same rule at a bar with
**no** break of structure, run through the same show/delete machinery.

Because every exit here is a stop 1R away and a target `k` R away, a driftless
walk wins with probability `1/(1+k)` - which is also the zero-cost break-even
rate. **`win - fair` is the setup's edge before costs**, and is compared
between the setup and its null.

### Decision rule

* **Primary cells:** 2 setups x 4 instruments x 12 timeframes = **96**, each at its
  stated exit - **`s0_t2`** (natural stop, 2R target) for both setups.
* **Statistic:** per-day t of net R (trades summed within a day, tested across
  days), one account per (instrument, timeframe, setup), one position at a time.
* **Dev pass** needs all three: net mean R > 0 at **t >= 3.27** (Sidak over 96,
  one-sided alpha 0.05); `win - fair` > 0; and `win - fair` above the matched
  null's for the same cell.
* **Validation**, only for a dev pass, and only after stating the expected
  validation t: same sign and t >= 2.0. The **test split is spent only on a
  validation survivor** whose expected test t is >= 2.5.
* **Exploratory:** the full 96 x 8 = 768 (cell x exit) grid, read at Sidak
  |t| >= 3.82, reported as counts beyond +/-2 and never used to select a cell
  for validation without a new registration.
* **Headline** figures pool dev and validation.

### Power, stated before the run

D1 has 4-10 sweeps and 22-28 blocks per instrument on dev; 4h about 40-55
sweeps. At those sizes a primary cell cannot reach t = 3.27 without a mean well
above 1R per trade, so **the verdict is decided at 1h and below** and the slow
rungs are read for sign and for the gradient, not for significance.

### Predictions

Prior studies in this directory make a specific forecast, written here so it
can fail:

* `level-interaction.md` (gold sweeps lose at the mid), `osler-round-numbers.md`
  (USDJPY big-figure sweeps *continue*), H14 in `discovery-program.md` (0 of the
  sweep-reclaim cells pass) and `engulfing-quadrant.md` (the stop run is worth
  -0.005 in strike rate) all say **SW forward return ~ 0 and `win - fair` ~ 0**,
  so SW loses roughly its `cost/R`, which is large at 1m because the wick stop is
  small.
* No study has tested order blocks. Their resting-limit entry at a level is the
  structure that the 2020-2023 minute-scale reversal (batch 6) would *favour*
  on dev, so a positive dev `win - fair` for OB that vanishes on validation is
  the regime artefact to watch for, not evidence.

---

## Results (appended 2026-09-12, after the run)

Every number below comes from `scripts/backtests/backtest_sweep_orderblock.py`
(the cell tables and ladders) or its `--diagnostics` pass (pooled splits, exit
curve, edge on cost, forward returns, break type). Output:
`reports/strategies/sweep_orderblock.json`,
`reports/strategies/sweep_orderblock_diagnostics.json`, and the order tapes in
`reports/strategies/sweep_orderblock_trades/`.

> **Verdict: rejected, both setups. No primary cell passes on dev (0 of 96), so
> nothing earned a validation shot; validation, read as a description, has no
> pass either. The test split was not spent.**
>
> Pooled over dev and validation at the stated exit (natural stop, 2R):
>
> | | trades | mean R | win | fair | win - fair | cost/R | cells > 0 | USD at 0.01 lot |
> | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
> | **liquidity sweep** | 126,787 | **-0.471** | 0.289 | 0.333 | -0.044 | 0.355 | 7 / 95 | -13,927 |
> | **order block** | 358,408 | **-0.528** | 0.275 | 0.333 | -0.058 | 0.285 | 6 / 96 | -40,645 |
>
> Across the full 768-combination exit grid on dev, 101 combinations are
> positive, **3 clear t = +2** (chance alone gives ~19) and 453 sit at t <= -2;
> none reaches the Sidak 3.82. On validation: 55 of 751 positive, **none at
> t = +2**, 464 at t <= -2.

### 1. The ladder: both converge on the coin from below, and never cross it

Pooled over the four instruments, stated exit. `edge` is the strike rate minus
the driftless value 1/3.

| tf | SW dev trades | SW dev edge | SW dev mean R | SW val edge | SW val mean R | OB dev trades | OB dev edge | OB dev mean R | OB val edge | OB val mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1m | 34,812 | -0.051 | -0.632 | -0.062 | -0.644 | 109,771 | -0.080 | -0.756 | -0.055 | -0.594 |
| 3m | 13,309 | -0.037 | -0.394 | -0.042 | -0.402 | 37,392 | -0.059 | -0.470 | -0.041 | -0.364 |
| 5m | 8,437 | -0.023 | -0.281 | -0.039 | -0.305 | 23,153 | -0.045 | -0.344 | -0.047 | -0.316 |
| 15m | 2,890 | -0.036 | -0.206 | -0.034 | -0.204 | 8,308 | -0.021 | -0.164 | -0.022 | -0.164 |
| 30m | 1,587 | -0.026 | -0.130 | -0.027 | -0.164 | 4,239 | -0.008 | -0.096 | -0.024 | -0.139 |
| 1h | 703 | -0.006 | -0.073 | -0.061 | -0.233 | 2,132 | -0.018 | -0.100 | -0.001 | -0.049 |
| 2h | 382 | +0.002 | -0.027 | -0.103 | -0.354 | 1,017 | -0.013 | -0.063 | +0.006 | -0.022 |
| 4h | 172 | +0.045 | **+0.137** | -0.109 | **-0.365** | 462 | -0.058 | -0.192 | -0.059 | -0.209 |
| 1d | 26 | -0.064 | -0.196 | -0.048 | -0.157 | 63 | +0.000 | -0.006 | -0.216 | -0.679 |

Cost per unit of risk falls from 0.53 (SW) and 0.42 (OB) at 1m to under 0.05
at 1h, and the loss falls with it - the shape of every price-action study in
this directory. What does not happen is the strike rate rising *past* 1/3 once
the cost is out of the way. The one pooled rung with a positive dev mean,
**sweeps at 4h (+0.137R, 172 trades, no cell above t = 1.3), is -0.365R on
validation**. D1 has 26 sweep and 63 block trades on dev and carries no
information, as the registration said.

### 2. No exit rescues either

Pooled dev + validation, every exit (`--diagnostics`):

| exit | SW mean R | SW win - fair | OB mean R | OB win - fair |
| --- | ---: | ---: | ---: | ---: |
| s0 / 1R | -0.485 | -0.071 | -0.527 | -0.090 |
| **s0 / 2R (stated)** | **-0.471** | **-0.044** | **-0.528** | **-0.058** |
| s0 / 3R | -0.467 | -0.037 | -0.532 | -0.046 |
| s0 / hold | -0.444 | - | -0.522 | - |
| s1 / 1R | -0.336 | -0.054 | -0.374 | -0.073 |
| s1 / 2R | -0.332 | -0.038 | -0.383 | -0.056 |
| s1 / 3R | -0.335 | -0.039 | -0.388 | -0.056 |
| s1 / hold | -0.323 | - | -0.372 | - |

The curve is flat across targets. The buffered stop (`s1`) looks better by
0.14R - and it is exactly the cost: it widens the risk, which cuts cost/R from
0.355 to 0.258 (SW) and 0.285 to 0.190 (OB), while the edge does not move.

### 3. Against the null: nothing survives normalising for the stop

On raw mean R the sweep "beats" its null by a mile (-0.471 against -1.523).
**That comparison is the trap documented in `engulfing-quadrant.md`**: a random
bar's wick is much closer to its close than a sweep bar's, so the null's 1R is
smaller and its cost/R is double (0.710 against 0.355). Spread moves both
barriers against the trade in proportion to cost/R, so strike rates must be
compared at equal cost. A trade-weighted fit of `edge = a + b * cost/R` across
cells gives the edge each variant would have at zero cost:

| intercept (edge at zero cost) | dev | validation |
| --- | ---: | ---: |
| sweep | -0.032 | -0.062 |
| sweep null | -0.076 | -0.060 |
| order block | -0.028 | -0.039 |
| order-block null | -0.032 | -0.048 |

* **The order block is its own null.** The zone built at a break of structure
  and the same zone built at a random bar with no break sit within 0.004 on dev
  and 0.009 on validation. The BoS/CHoCH condition carries nothing - and the
  indicator's own labels confirm it: CHoCH blocks lose -0.584R and BoS blocks
  -0.583R on dev (edge -0.064 and -0.068), -0.461R and -0.457R on validation.
* **The sweep beat its null on dev by 0.044 and lost to it on validation by
  0.002.** That is the one dev-only hint in the study, and it is the pattern
  `discovery-program.md` already explained: minute-scale reversal is a
  2020-2023 property that the post-Q4-2023 microstructure no longer pays.
* All four intercepts are negative. The machinery - fills on the far side of
  the spread, market stops, ties to the stop - costs a few points of strike rate
  that `cost/R` does not capture, and the null pays it too, which is why the
  comparison is between intercepts and not against zero.

### 4. What the signals predict, with every barrier removed

Forward return from the fill mid, signed in the trade's direction, pooled over
the instruments, per-day t:

| | 1m fwd1 | t | 1m fwd5 | t | 5m fwd5 | t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep, dev | +0.059 | 2.1 | +0.009 | 0.2 | -0.112 | -0.5 |
| sweep null, dev | +0.006 | 0.4 | +0.040 | 1.2 | +0.036 | 0.2 |
| sweep, validation | -0.074 | -2.4 | **-0.147** | **-2.5** | -0.344 | -1.2 |
| order block, dev | **-0.198** | **-11.4** | -0.195 | -7.8 | -0.302 | -2.1 |
| order-block null, dev | -0.089 | -7.4 | -0.070 | -3.4 | -0.078 | -0.7 |
| order block, validation | -0.031 | -1.2 | **-0.110** | **-3.1** | -0.388 | -2.1 |
| order-block null, validation | -0.003 | -0.2 | -0.005 | -0.2 | -0.503 | -3.2 |

* **The sweep reversal is a fraction of a basis point in 2020-2023 and has
  turned into continuation since.** On dev the 1m sweep reverses by 0.06 bps
  over the next bar - real (t 2.1) and a tenth of a round turn. On validation the
  swept side keeps going (-0.147 bps at 5 bars, t -2.5). USDJPY was continuation
  already on dev (5m fwd5 -0.78 bps, t -3.2), the same sign `osler-round-numbers.md`
  found for USDJPY big-figure sweeps.
* **A resting limit at an order block is adversely selected.** After the fill
  price keeps running *through* the block: -0.198 bps over the next minute at
  t = -11.4 on dev, more than twice the null zone's -0.089. The orders that fill
  are disproportionately the ones a move ran over; the moves that bounce short of
  the near edge never fill. On validation the 1m block is still -0.110 bps at
  5 bars (t -3.1) against a null of -0.005.

### 5. The orders themselves

Block orders pooled over instruments and timeframes: 65.1% fill on dev (65.6%
validation), 17.8% are cancelled because two newer blocks hid them, 16.3%
expire at the 100-bar cap, and **0.8% are deleted by a close through the far
edge before filling** - the indicator's deletion rule almost never pre-empts
the order, because price has to pass the near edge first. The 100-bar cap binds
on one order in six; since fills cluster early (median 17 bars after creation)
and the setup loses at every fill delay, a longer life would add losing trades.

Sweep orders all fill at market. The median sweep-bar wick stop is 2.5 bps
away, which is why 1m sweeps pay 0.53R of cost before the trade has moved.

Swap at the measured overnight basis moves the 4h and D1 cells by at most
0.05R and flips no cell that was distinguishable from zero. Per-day t on the
D1 cells is not meaningful: with 3-5 trades that all lose exactly 1R the
standard deviation collapses, which is where the four- and five-digit
t-statistics in the validation D1 rows of the run log come from.

### 6. Against the predictions

* **SW: forward return ~0 and `win - fair` ~0 on dev, loss ~cost/R** - held.
  Dev 1m fwd5 is +0.009 bps; the pooled edge is -0.044 and the zero-cost
  intercept -0.032; the loss tracks cost/R down the ladder.
* **OB: a positive dev edge that vanishes on validation** - did not happen. The
  dev edge was negative too; the 2020-2023 reversal that should favour a resting
  limit is swamped by the adverse selection of the fill.

### 7. What would have to be true

1. **The sweep reverses by more than a round turn within the trade's life.** It
   reverses by 0.06 bps on dev at 1m, a tenth of the round turn, and continues
   on validation.
2. **A block fill is followed by a bounce.** It is followed by continuation
   through the block, twice as strongly as a random zone of the same shape.
3. **The structure break selects better zones.** The zero-cost edge of a
   break-of-structure block and a no-break block differ by 0.004 on dev; CHoCH
   and BoS differ by 0.001R.
4. **Some exit extracts what the signal has.** There is no signal to extract,
   and the 8-exit curve is flat across targets.

## Reproducing

```bash
python scripts/backtests/backtest_sweep_orderblock.py                # dev + validation, setups + null (~5 min, 4 workers)
python scripts/backtests/backtest_sweep_orderblock.py --from-tapes   # re-summarise, no tape walk
python scripts/backtests/backtest_sweep_orderblock.py --diagnostics  # sections 2-5
python -m pytest tests/test_sweep_orderblock.py
```

**The test split (2025-07-01 onward) remains unspent.**

---

## Addendum A - order block to the opposite block (pre-registered 2026-09-12, before any number for it)

The owner's reading of the trade, which the main study did not run: **enter at
the start of the block, stop at its other end, take profit at the opposite-side
order block.** Entry and stop are the main study's `s0` exactly - a resting
limit at the near edge, stop at the far edge, risk one ATR. Only the target is
new, and it is what makes this a different bet: its distance is set by where
the chart's other blocks sit, not by a multiple of the risk.

```text
OB-T  order block, opposite-block target
    Entry, stop, order life, show/delete rules: as OB above.
    Target:   at order placement (the block's creation bar, after the script's
              update), the nearest opposite-side block the chart shows - for a
              long, the lowest shown bearish block whose bottom is above the
              entry; for a short, the highest shown bullish block whose top is
              below it. Fixed for the life of the trade.
              obn = that block's near edge (the first price where it starts)
              obf = its far edge (price must cross the whole block)
    None:     if the chart shows no opposite block beyond the entry, no trade.
              Its share is reported.
    Shown:    the script's display rule exactly - the newest 2 per side,
              including a hidden block that resurfaces when a newer one is
              deleted.
    Null:     the matched no-break zones, given the target from the same real
              blocks on the chart.
    Life:     60 bars after the fill, as before; s1 (stop 0.5 ATR beyond the far
              edge) reported beside s0.
```

**Fair coin, per trade.** The target is `k` R away with `k` varying trade by
trade, so the driftless strike rate is `1/(1+k)` per trade and the comparison
is the realised strike rate against its mean.

**Decision rule.** Primary cells: 4 instruments x 12 timeframes = **48**, at
**`s0_obn`**. Dev pass needs net mean R > 0 at per-day **t >= 3.07** (Sidak over
48, one-sided 0.05), strike rate above its fair coin, and above the null's
margin in the same cell. Validation only for a dev pass, after stating its
expected t; test only for a validation survivor. `obf` and `s1` are secondary.

**Multiplicity stated plainly.** This is a second draw from the family the main
study rejected: same entries, same stops, same fills. A pass here would have to
be read against the 96 cells already spent.

**Prediction.** Entry and stop are unchanged, so the adverse selection of the
fill (section 4 above: price runs through the block after it fills) and the
cost per unit of risk (0.42R at 1m, under 0.05R from 1h) carry over unchanged.
The only route to an edge is the target: if opposite blocks are where price
turns, a target placed at one is reached more often than its fair coin says.
The forecast from the main study is that it is not - strike rate at or below
`1/(1+k)`, loss tracking cost/R down the ladder.

### Addendum A - results (appended 2026-09-12, after the run)

Script: `scripts/backtests/backtest_sweep_orderblock.py -f ob --primary s0_obn`.
Output: `reports/strategies/sweep_orderblock_ob.json`, tapes
`reports/strategies/sweep_orderblock_trades/sob_*_ob.parquet`, log
`reports/strategies/sweep_orderblock_ob.log`.

> **Verdict: rejected. 0 of 48 primary cells pass on dev; validation has no
> pass either. The test split was not spent.**
>
> Pooled over dev and validation at `s0_obn`: **315,565 trades, -0.545R per
> trade**, strike rate **0.143 against a per-trade fair coin of 0.191**
> (-0.048), 4 of 95 cells positive, **-$38,540** at 0.01 lot. On the full grid,
> 2 of 576 dev combinations clear t = +2 (chance gives ~14) and 349 sit at
> t <= -2; on validation 0 of 560 and 332.

**The target is far, so the bet needs a low strike rate - and does not get
it.** The nearest opposite block sits a median **5.1R** beyond the entry on dev
(interquartile 3.2-8.1R) and 4.7R on validation, so a driftless walk wins about
one trade in five. Only 5.6% of orders (dev) have no opposite block on the chart
to aim at, rising from 5% at 1m to 32% (dev) and 58% (validation) at D1.

The ladder at `s0_obn`, pooled over the instruments; `null` is the no-break
zones' margin over their own fair coin, with the same targets:

| tf | dev trades | cost/R | win | fair | edge | null | mean R | val edge | val null | val mean R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1m | 97,292 | 0.428 | 0.126 | 0.183 | -0.057 | -0.043 | -0.775 | -0.049 | -0.041 | -0.626 |
| 2m | 49,080 | 0.296 | 0.137 | 0.188 | -0.051 | -0.043 | -0.575 | -0.046 | -0.043 | -0.473 |
| 3m | 33,101 | 0.239 | 0.141 | 0.190 | -0.049 | -0.040 | -0.480 | -0.045 | -0.039 | -0.387 |
| 5m | 20,404 | 0.182 | 0.151 | 0.191 | -0.041 | -0.036 | -0.328 | -0.049 | -0.041 | -0.357 |
| 10m | 10,326 | 0.125 | 0.158 | 0.193 | -0.035 | -0.036 | -0.196 | -0.036 | -0.038 | -0.222 |
| 15m | 7,141 | 0.097 | 0.164 | 0.193 | -0.029 | -0.020 | -0.162 | -0.028 | -0.028 | -0.168 |
| 20m | 5,524 | 0.082 | 0.172 | 0.198 | -0.026 | -0.030 | -0.125 | -0.036 | -0.023 | -0.179 |
| 30m | 3,595 | 0.064 | 0.181 | 0.201 | -0.019 | -0.028 | -0.095 | -0.031 | -0.036 | -0.201 |
| 1h | 1,801 | 0.043 | 0.189 | 0.214 | -0.025 | -0.013 | -0.047 | -0.023 | -0.006 | -0.105 |
| 2h | 832 | 0.029 | 0.194 | 0.213 | -0.020 | -0.016 | -0.088 | -0.038 | -0.020 | -0.138 |
| 4h | 379 | 0.020 | 0.137 | 0.204 | -0.067 | -0.003 | -0.289 | -0.070 | -0.031 | -0.281 |
| 1d | 38 | 0.008 | 0.132 | 0.215 | -0.083 | -0.078 | -0.169 | -0.197 | -0.042 | -1.008 |

Three findings:

* **The strike rate sits below its fair coin on every rung of both splits.** As
  cost/R falls from 0.43 to under 0.05 the loss shrinks, as it always does here,
  and the edge closes toward zero from below without crossing it. No timeframe
  is slow enough.
* **Opposite blocks are not a magnet.** A target at an opposite block is reached
  *less* often than a driftless walk reaches a level that far away, and the
  setup's margin is below the null zones' in 18 of the 24 rungs. The chart's
  other blocks carry no information about where price goes next, which is the
  one thing this reading needed them to.
* **No variant of it helps.** Pooled over both splits:

  | exit | mean R | win | win - fair |
  | --- | ---: | ---: | ---: |
  | far-edge stop, opposite block's near edge (**stated**) | **-0.545** | 0.143 | -0.048 |
  | far-edge stop, opposite block's far edge | -0.544 | 0.102 | -0.043 |
  | buffered stop, near edge | -0.393 | 0.188 | -0.063 |
  | buffered stop, far edge | -0.393 | 0.138 | -0.061 |
  | *main study: far-edge stop, fixed 2R* | *-0.528* | *0.275* | *-0.058* |

  The buffered stop again looks better only because it widens the risk and
  cuts cost/R (0.290 to 0.193); its margin over the coin is *worse*. The
  opposite-block target is no better than a fixed 2R on the same entries.

What the addendum adds to the main study: the entry is unchanged, so the
adverse selection of the fill is unchanged (section 4), and a target set from
the chart's own structure does not recover it. **The test split remains
unspent.**

---

## Addendum B - take the other side (pre-registered 2026-09-13, before any number for it)

The owner's question after the rejection: if the trades lose more often than
chance, does the opposite trade win? Written here before any fade was resolved.

### What "the opposite" means

```text
mirror (PRIMARY)  the other side of the same bracket, at the same prices.
    Direction reversed. The original target becomes the stop; the original
    stop becomes the target. Levels are fixed from the original reference
    price (the sweep bar's close; the block's near edge).
    SW-fade   after a swept high: long at market at the close, target the
              sweep wick, stop where the short's kR target was.
    OB-fade   when price reaches the block's near edge: a stop order in the
              direction of the move INTO the block (sell stop for a bullish
              block), target the block's far edge, stop where the long's
              target was. Order life, hide and delete rules unchanged.
    Exits     mirrors of s0/s1 x 1R/2R/3R; for blocks also the mirror of the
              opposite-block target (obn, obf). No "hold" - its mirror has no
              stop.
flip (secondary)  reverse direction, keep the usual shape: stop the same
    distance beyond the entry on the other side, targets 1R/2R/3R and hold.
```

A mirrored target sits `1/k` of its stop away, so its driftless strike rate is
`k/(1+k)` per trade (0.667 for the mirror of a 2R trade), computed per trade
from the actual fill.

### Prediction, from arithmetic on the main study

Before costs a mirrored bracket earns exactly what the original lost; after
costs it pays the round turn again, and the block's entry becomes a stop order
that pays slippage where the original limit did not. Pooled main-study numbers
at `s0_t2`, in the original's R:

| | original net | cost/R | original before cost | mirror before cost | mirror after cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| sweep | -0.471 | 0.355 | -0.116 | +0.116 | **about -0.24** |
| order block | -0.528 | 0.285 | -0.243 | +0.243 | **about -0.04 to -0.10** |

The approximation is not exact - the mirror fills on the other side of the
spread - which is why it is run rather than computed. It says the sweep fade
loses, and the block fade is the one place an edge could appear, most plausibly
at 1m-5m where the original lost most before cost (1m: -0.335R) and where
section 4 found price running through a filled block at t = -11.4.

### Decision rule

* **Primary:** mirror at `s0_t2` (the other side of the stated trade),
  2 setups x 4 instruments x 12 timeframes = **96 cells**. Dev pass: net mean
  R > 0 at per-day **t >= 3.27**, strike rate above its per-trade fair coin, and
  above the faded matched null's margin in the same cell.
* **The validation split is not clean for this.** The fade was chosen after the
  original's dev *and* validation results were read, and a mirrored trade's
  pre-cost P&L is the negative of the original's. Validation is reported, but
  it is not evidence. A dev pass that holds on validation earns one run on the
  locked test split, and only if its expected test t is >= 2.5, stated first.
* **Multiplicity:** the third draw from this family (96 + 48 cells already
  spent), on the same entries.

### Addendum B - results (appended 2026-09-13, after the run)

Script: `scripts/backtests/backtest_sweep_orderblock.py --fade mirror` and
`--fade flip`. Output: `reports/strategies/sweep_orderblock_fade-{mirror,flip}.json`,
logs beside them, tapes `reports/strategies/sweep_orderblock_trades/*_fade-*.parquet`.
A normal-mode re-run of EURUSD 1h/4h/D1 on validation reproduced both saved
tapes with zero differences over 904 trades before either fade was launched.

> **Verdict: rejected, both readings. 0 of 96 primary cells pass on dev for
> the mirror and 0 of 96 for the flip. The test split was not spent.**
>
> Pooled over dev and validation, at 0.01 lot, stated exit (`s0_t2` or its mirror):
>
> | | original | mirror (other side) | flip (reversed) |
> | --- | ---: | ---: | ---: |
> | liquidity sweep | -$13,927 | **-$12,280** | **-$9,715** |
> | order block | -$40,645 | **-$42,107** | **-$36,675** |
> | sweep: win / fair coin | 0.289 / 0.333 | 0.622 / 0.711 | 0.291 / 0.333 |
> | block: win / fair coin | 0.275 / 0.333 | 0.619 / 0.701 | 0.298 / 0.333 |
>
> Mean R is -0.471 / -0.300 / -0.474 (sweep) and -0.528 / -0.292 / -0.497
> (block); the mirror's R is measured on a stop twice as wide, so compare the
> dollars across columns, not the R.

**A trade and its exact opposite both land below their fair coin.** That is the
finding. A directional edge makes one side win what the other loses; here both
sides lose, which is only possible if the shortfall is mechanical - the spread
both fills cross, slippage on market legs, and the stop-first tie rule - rather
than a direction that can be reversed.

**Both arithmetic predictions were too kind.** The sweep mirror was forecast at
about -0.12 of its own R and came in at -0.300; the block mirror at -0.02 to
-0.05 and came in at -0.292. The forecast assumed the mirror's pre-cost result
is the negative of the original's. It is not, because **the opposite of a
resting limit is not a stop order at the same price**: a sell stop at the
block's near edge triggers when the bid reaches it, half a spread before the
original buy limit would fill, so it trades on touches the limit never saw
(116,619 dev 1m block trades against 109,771) and pays slippage on every entry.
At 1m that half spread and slippage are a large share of a one-ATR-wide risk.

**The ladder.** Both readings lose on every rung from 1m to 1h in both splits
and converge on zero as cost/R falls, as the originals did. The one rung
positive in both splits for both readings is **order blocks at 4h**:

| order blocks, 4h, `s0_t2` | dev trades | dev mean R | val trades | val mean R | pooled | per-day t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 463 | +0.049 | 182 | +0.071 | +0.055 | 2.11 (2.06 with swap) |
| mirror, matched null | 567 | +0.012 | 232 | +0.037 | | |
| flip | 459 | +0.061 | 180 | +0.273 | +0.121 | 2.16 |

It is recorded as an observation, not a candidate:

* it was picked after looking - the only rung of 24 positive in both splits -
  and t = 2.1 is below the ~2.8 that selecting from 24 requires, let alone the
  registered 3.27; the two readings share their entries and are not two pieces
  of evidence;
* neither split is significant alone (mirror t = 1.62 dev, 1.38 validation);
* **the test split could not confirm it if it were real**: ~14 months of 4h
  blocks is ~140 trades, an expected t of about 1.0 at this effect size.

**Grid counts.** Mirror: 6 of 768 dev combinations at t >= 2, 4 of them D1 cells
of 4-10 trades; 28 of 740 on validation, 20 of them D1 cells of 3-5 trades.
Flip: 2 of 768 dev; 21 of 746 validation, the 6 above 3.82 all D1 cells of 3-4
trades. A D1 cell whose handful of trades all win the same amount has a
near-zero standard deviation and a per-day t in the thousands; the mirror run
printed four such cells as validation "passes" (t 91 to 3,011). They were never
eligible - validation counts only after a dev pass - and the script now
requires 30 trading days before a cell can be listed as a pass.

**The test split (2025-07-01 onward) remains unspent.**
