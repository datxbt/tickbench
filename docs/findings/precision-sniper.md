# Precision Sniper (WillyAlgoTrader) - evaluation

Subject: a TradingView confluence indicator (Pine v6 v1.4.0, © Willy /
WillyAlgoTrader), turned into trades and run tick by tick on EURUSD, USDJPY,
XAUUSD and USTEC at **twelve timeframes from 1m to D1**, across all six of its
parameter presets and both settings of its higher-timeframe filter.

The indicator scores, grades and draws; it does not trade. Every trading
choice below was made - and fixed - before any outcome was computed.
Implementation: `src/qlab/strategies/precision_sniper.py`. Driver:
`scripts/backtests/backtest_precision_sniper.py`. Tests:
`tests/test_precision_sniper.py`.

---

## Pre-registration (written 2026-09-12, before any grid result was computed)

### What had been seen first

Honesty about the order of operations, because it bears on how the numbers
below should be read:

1. **Signal counts** from a bars-only dry run
   (`--dry-run --splits dev -s USTEC`): 1,523,618 signals across twelve
   timeframes, six presets and two HTF settings. No tick resolved.
2. **A smoke test** while building the resolver: USTEC, 5m, `Default` preset,
   June 2023 only, 253 fills - one cell of one month out of a grid of 576.
   It was run to prove the tick path worked (it found an ordering bug in the
   summary print), and its numbers are not evidence for or against anything.
   They are reproduced here so that nothing is retro-fitted: mean R was
   negative at every single target, win rates ran below their fair rates, and
   cost was 10.9% of risk.

No other outcome has been read. The grid below is specified in full before it
runs, and the verdict rule is fixed before the first cell is seen.

### The engine, as ported

```text
Signal    ta.crossover(emaFast, emaSlow), price above both EMAs, RSI < 75
          (mirrored for shorts), and confluence score >= the preset minimum.
Score     Ten factors in the script; eight here. This feed quotes bid and ask
          with no size, so the volume-surge and VWAP factors do not exist and
          the script's own adaptive scale applies: max 8.0, not 10.0, with the
          minimum score rescaled the same way (Default 5 of 10 -> 4.0 of 8).
Grades    A+ >= 80% of max, A >= 65%, B >= 50%, C below - the script's bands,
          on the same adaptive scale.
Account   Stop-and-reverse, no pyramiding. This is what lastDirection is: it
          blocks a same-direction signal while a position is open and takes an
          opposing one, closing the open trade at the opposing signal's price
          (the script's own backtest Case 2).
```

### The grid

| axis | levels | n |
| --- | --- | ---: |
| instrument | EURUSD, USDJPY, XAUUSD, USTEC | 4 |
| timeframe | 1m 2m 3m 5m 10m 15m 20m 30m 1h 2h 4h 1d | 12 |
| preset | Scalping, Aggressive, Default, Conservative, Swing, Crypto 24/7 | 6 |
| HTF filter | the tooltip's ladder, or off | 2 |
| stop | `atr`, `struct`, and each with High-vol widening (`_w`) | 4 |
| exit | 0.5R 1R 1.5R 2R 3R 4R, hold, and four managed ladders | 11 |

`Auto` is not a seventh preset, it is a timeframe-to-preset map, so it is read
off the result rather than run. `Custom` is the input defaults, which are
`Default` exactly.

**576 signal cells**, one-sided Šidák t ≥ 3.75; **25,344 grid cells**,
one-sided Šidák t ≥ 4.61. Both thresholds are computed in the driver and
printed before any cell is.

The script's four switches - grade filter, hide-C, volatility mode, minimum
score - are recorded per signal and applied to the resolved tape afterwards,
so all 72 of their combinations are priced without a second tick walk.

### Exits, and why they are swept

The script nominates one: three R-multiple targets (1R/2R/3R) with the stop
stepping to breakeven at TP1, to TP1 at TP2, to TP2 at TP3, and a full close
at TP3. That rule is simulated tick by tick and is the **primary**,
`struct_pL123`. Two ladders either side of it (1/1.5/2 and 1.5/3/4.5) and the
pre-v1.4.0 runner are priced too, along with plain targets from 0.5R to 4R and
a stop-or-clock hold - so the verdict never rests on one arbitrary exit.

Positions die after 60 bars of their own timeframe; the script has no clock, so
the time-exit rate is reported for every cell.

### The control

A size-matched mechanics-only null: the same number of ordinary bars of the
same timeframe, drawn by a deterministic hash of the timestamp, entered at
market on the bar's close with the same ATR and structure stops and run
through the identical exit grid, direction a hash coin. It is what separates
*the confluence engine picks bars* from *a 1R stop and an R-multiple target
does this to any bar*.

### Costs

The measured cost model (`docs/cost-model.md`), fitted on the same split the
cells are drawn from: spread through the fill prices, commission per side,
slippage on market entries and stops but not on limit targets. Overnight swap
is charged from the measured basis for the exits that can hold across a roll.

### The verdict rule, fixed now

A cell **passes** only if, on dev, it has mean R > 0 with per-day t ≥ 3.75
(the cell-count Šidák), beats its own matched null, and then repeats with the
same sign on validation. Anything that survives both is a candidate; the
`test` split is spent only on a survivor, and stays locked otherwise.

Anything else is a rejection, and the write-up says so.

---

## Results

**Rejected.** 17,780,538 signals resolved tick by tick - 8,894,154 setups and
8,886,384 matched-null bars - across 552 cells and 44 exits per cell, on dev
and validation. Nothing passes. The `test` split was not touched.

The run: `python scripts/backtests/backtest_precision_sniper.py --filters`,
then `--diagnostics` for the tables that compare against the null.

### The whole grid, in one number each

| | dev | validation |
| --- | ---: | ---: |
| cells x exits with trades | 24,244 | 24,102 |
| mean R > 0 | 5,347 | 6,788 |
| **t >= Šidák (4.61)** | **0** | **2** |
| t <= -Šidák | 8,968 | 6,211 |

The two on validation are the same five trades counted twice: USDJPY daily,
`Aggressive`, no HTF, a 4R target, five trades of which four reached it, under
two stop keys that coincide because none of the five was a High-volatility
bar. They do not appear on dev, and the pre-registered rule requires both.

### Mean R is a readout of cost, not of skill

`struct_pL123` - the script's own stop and its own managed exit - pooled over
instruments, presets and HTF settings:

| tf | trades (dev) | cost/R | dev mean R | null | validation mean R | null |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1m | 1,973,876 | 0.250 | −0.3467 | −0.3476 | −0.3123 | −0.3156 |
| 2m | 964,252 | 0.169 | −0.2601 | −0.2616 | −0.2262 | −0.2353 |
| 3m | 629,824 | 0.135 | −0.2149 | −0.2148 | −0.1812 | −0.1826 |
| 5m | 367,224 | 0.099 | −0.1546 | −0.1651 | −0.1142 | −0.1470 |
| 10m | 177,629 | 0.065 | −0.1009 | −0.1093 | −0.0570 | −0.1008 |
| 15m | 114,729 | 0.052 | −0.0819 | −0.0773 | −0.0371 | −0.0689 |
| 20m | 84,588 | 0.044 | −0.0679 | −0.0764 | −0.0320 | −0.0866 |
| 30m | 56,534 | 0.035 | −0.0616 | −0.0546 | −0.0114 | −0.0483 |
| 1h | 28,714 | 0.024 | −0.0105 | −0.0352 | +0.0117 | −0.0736 |
| 2h | 15,081 | 0.017 | +0.0379 | +0.0062 | +0.0090 | −0.0799 |
| 4h | 7,794 | 0.012 | +0.0665 | +0.0785 | +0.0318 | +0.0091 |
| 1d | 544 | 0.017 | +0.1019 | +0.0591 | +0.1743 | −0.1558 |

Mean R rises monotonically with the timeframe and crosses zero exactly where
the round turn stops eating about 2% of the stop distance. The TP3 strike rate
does not move with it: it sits between 0.099 and 0.145 at **every** timeframe
on both splits. A rule whose payoff varies by half an R while its hit rate is
flat is not changing its skill across timeframes; its costs are changing.

Fitting that directly, trade-weighted across all 552 cells:

| split | variant | a (mean R at zero cost) | b |
| --- | --- | ---: | ---: |
| dev | setup | −0.0292 | −1.288 |
| dev | null | −0.0264 | −1.206 |
| validation | setup | +0.0106 | −1.603 |
| validation | null | −0.0091 | −1.420 |

At zero cost the engine returns nothing, and neither does a coin flip on a
random bar with the same stop. The intercept changes sign between the splits;
the difference between setup and null (−0.003 on dev, +0.020 on validation)
does not hold its sign either.

### The signal itself has no direction in it

Forward mid return from the fill, no barrier and no cost - the measurement no
exit choice can affect. Across 24 timeframe-by-split cells the largest |t| is
**3.16**, on dev 1m over five bars, and it is **negative** (−0.058 bps). Every
other cell is inside ±2.5. There is no horizon at which these signals predict
the next move.

### Every exit, and every stop

Pooled dev + validation, all 44:

| stop | best exit | mean R | null | worst exit | mean R |
| --- | --- | ---: | ---: | --- | ---: |
| `atr` | `prL123` | −0.3150 | −0.3001 | `t05` | −0.3457 |
| `atr_w` | `prL123` | −0.3118 | −0.2947 | `t05` | −0.3400 |
| `struct` | `prL123` | −0.2446 | −0.2528 | `t2` | −0.2641 |
| `struct_w` | `prL123` | −0.2423 | −0.2488 | `t2` | −0.2610 |

The whole 44-cell spread is 0.10R wide and entirely below zero. The structure
stop beats the ATR stop by about 0.08R for one reason: it is wider, so the
round turn is a smaller share of it (cost/R 0.17 against 0.25). On the ATR
stops the engine is *worse* than its null; on the structure stops it is better
by half a hundredth of an R. At 0.01 lots the script's own configuration lost
**$621,082** over 6.02M trades.

Strike rates against their fair rates (`1/(1+k)` for a k-R target against a 1R
stop, the driftless first-passage null) are negative at every rung: −0.079 at
0.5R, −0.065 at 1R, −0.072 at 2R, −0.093 at 4R. The shortfall tracks cost/R,
which is what a driftless walk with a spread does.

### The script's four switches

All 72 combinations of grade filter, hide-C, volatility mode and minimum
score, at the primary exit:

| | dev | validation |
| --- | ---: | ---: |
| best (`A+ Only`, vol `Off`) | −0.2264 | −0.1894 |
| the script's defaults | −0.2631 | −0.2271 |
| worst (`All`, `Skip Signals`) | −0.2726 | −0.2364 |

Every combination loses. The best is better than the worst by 0.046R against a
cost of 0.173R, and `Skip Signals` - the one switch v1.4.0 added - is the
worst of the four on both splits.

Three of the four barely do anything, and the reasons are structural rather
than empirical:

1. **The minimum score cannot bind at its default.** Trade counts are
   *identical* at min-score 0, 3, 4 and 5 (4,412,356 on dev); only 6 and 7
   remove anything. A buy signal requires `ta.crossover(emaFast, emaSlow)`,
   which awards its 1.0, and `close > emaFast`, which awards its 0.5 - so no
   signal can score below 1.5, and the observed median is 5.83 against a
   `Default` threshold of 4.0.
2. **Hide C-Grade removes 0.6% of signals.** C requires a score under 50% of
   max, and the observed grade mix is A+ 32.6%, A 45.6%, B 21.2%, C 0.6%.
3. **With the HTF filter at its default empty setting, the HTF factor is
   awarded on 0.000 of signals** - the heaviest single factor, 1.5 of 8
   points, deterministically dead. `request.security("", ema[1])` reads the
   *chart's previous bar*, and on a crossover bar the previous bar's EMA
   relation is by construction the opposite of the cross. Setting a real HTF
   revives it, at which point it agrees with the trade direction 48-51% of the
   time - a coin. Median score goes 5.52 -> 6.1 and the A+ share 18% -> 46%
   purely from switching the input on.

### What the Auto preset selects

`Auto` maps 5m and below to `Scalping`, which is where cost/R is worst. Its
own cells are the worst in the study: EURUSD 1m at cost/R 0.366, mean R
**−0.4862** at t −57.7 on dev and −0.5664 at t −33.5 on validation. The
positive cells are all 2h and above, where `Auto` selects `Conservative` and
`Swing` - 20 to 165 trades per cell, |t| below 2.1, and signs that do not
survive the split change (EURUSD 2h: +0.006 on dev, +0.337 on validation;
XAUUSD 4h: +0.241 on dev, +0.421 on validation on 20 trades).

### What would change the verdict

Nothing in the parameter space: it was swept. Two honest caveats, neither of
which rescues it:

- **Two of the ten factors were untestable.** This feed has no volume, so the
  volume-surge and VWAP factors do not exist and the script's own adaptive
  scale ran the engine at max 8.0. On a symbol with real volume they might
  carry information - though the volume factor is added to the bull *and* bear
  scores identically, so it cannot discriminate direction whatever it does.
- **The corpus is four instruments of FX, metal and an index CFD.** A
  `Crypto 24/7` preset was run, but not on crypto.

Neither touches the central result, which is not about any one factor: at zero
cost the rule returns zero, its forward returns carry no direction at any
horizon, and it matches a coin flip on a random bar with the same stop
geometry to within a hundredth of an R. The indicator's own stated purpose -
that confluence scoring and grading identify higher-quality entries - is the
thing that does not replicate. Grade A+ signals are not better than grade B
signals by enough to matter, because the grade is mostly a readout of whether
one 1.5-point factor was switched on in the inputs.

### Artifacts

- `src/qlab/strategies/precision_sniper.py` - the port.
- `scripts/backtests/backtest_precision_sniper.py` - the sweep and the tables.
- `tests/test_precision_sniper.py` - 40 tests.
- `reports/strategies/precision_sniper.json` - every cell.
- `reports/strategies/precision_sniper_trades/` - the order tapes (gitignored;
  rebuild with the driver, re-summarise with `--from-tapes`).
