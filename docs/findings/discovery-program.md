# Discovery program: tail-conditioned structure

A running research log, not a single study. Each batch is pre-registered here
**before** its script is run, then its results are appended beneath it whatever
they say. Hypotheses carry an ID and a parent, so a child mutated from a failure
is visibly one more draw from the same well, not an independent discovery.

## Why this corner

Twenty-five studies in this directory establish two constraints that bound
where an edge can still be:

- **Minute-level predictability is real and too small.** Linear minute
  autoregression reaches t = -20 to -29 on EURUSD and USTEC and +19.8 on USDJPY,
  and its implied edge is 0.19x-0.48x of a round turn everywhere.
- **Daily samples are too short.** Dev holds ~980 sessions; a true Sharpe of 0.8
  gives t ~ 1.6 there.

A linear coefficient averages over every minute. A fixed cost does not scale
with the move, but a conditional edge can: if the response to a move is
`beta(|z|) * move`, then conditioning on large `|z|` raises the expected edge
per trade while the round turn stays put. The previous tests were linear or
unconditional (minute AR, cross-asset R-squared, hour-of-day shape), so the
tails - where the edge/cost ratio could flip - are untested. That is the corner
this program searches.

## Protocol

- **Splits.** Discovery on `dev` only. A hypothesis reaches `validation` only if
  its pre-registered primary cell clears the batch's Sidak threshold on dev
  **and** its date-shifted placebo. `test` is spent only on a survivor of both.
- **Unit.** Forward return in bps from the event bar's closing quote, reported
  three ways (`qlab.eventstudy`): mid, bid/ask fills + commission, and net of
  measured slippage. Inference uses the **per-day** t (events summed within a
  day, tested across days), because events cluster.
- **Normalisation.** A move is scored `z = r_W / (sigma_t * sqrt(W) * d_b)`:
  `sigma_t` the trailing 1,440-minute standard deviation of 1-minute log returns
  ending before the move window, `d_b` the dev-sample RMS diurnal factor for the
  bar's 15-minute UTC bucket. The diurnal factor is an in-sample normalisation
  (a scale, not a signal), which is stated here rather than hidden.
- **Classification** of any result: OBSERVATION -> PREDICTIVE (mid, per-day
  |t| past threshold) -> EXPLOITABLE (net positive) -> ROBUST (survives
  validation, placebo, perturbation, regimes) -> PRODUCTION_CANDIDATE (and the
  test split, and executable on ticks).

## Registry

| ID | parent | family | status |
| --- | --- | --- | --- |
| H01 | - | large-move response | **rejected** |
| H02 | H01 | scheduled x unscheduled interaction | **rejected** |
| H03 | - | cross-market shock transmission (catch-up) | **rejected** |
| H04 | - | compression -> expansion | **rejected** (direction); expansion itself an OBSERVATION |
| H05 | H01, H04 | USDJPY intraday continuation | **rejected** on validation; kept as an OBSERVATION |
| H06 | H03 | plain cross-market transmission, pooled | **rejected** |
| H07 | H05 | USDJPY continuation, up-moves only | **not tested** - dev-selected, reversed on validation (descriptive), no clean data left |
| H08 | - | spread-shock reversal (liquidity withdrawal) | **rejected** - sign is backwards |
| H09 | H01 | activity-conditioned reversal (Campbell-Grossman-Wang) | **rejected** |
| H10 | USTEC #15 | weekend reopen reversal, pooled over four instruments | **rejected** |
| H11 | H08 | spread-shock *continuation* (adverse selection), pooled | **not taken** - expected validation t 1.26 < 2.5 |
| H12 | USTEC #8, #9; H01-H11 | cost-aware nonlinear forecast, 4 instruments, 30 m | **rejected** - no out-of-sample skill; mid +0.3 bps < commission floor |
| H13 | H12, H05, H11 | H12 at 120 m | **rejected** - dev-holdout net -0.52 bps, t -2.06 |
| H14 | H01, H03, H04, sweeps, ORB | conditional micro-structure screen, 81,040 cells | **rejected** - 0 cells pass stage A; reversal structure at the mid replicates (diagnostic) |
| H15 | H14 diagnostic | portfolio of the 23 replicating cells that clear cost | **not taken** - expected validation t 1.64 < 2.5 |
| H16 | H15 | H15 on tick fills (execution, not signal) | **rejected** on validation - net -0.44 bps/trade, t -3.57; mid +0.09 |

---

## Batch 1 (pre-registered 2026-09-11, dev only)

Script: `scripts/research/discovery_batch1.py`. Output: `reports/discovery/batch1/`.
Primary cells: 4 (H01) + 4 (H02) + 12 (H03) + 4 (H04) = **24**, so a primary
cell needs per-day **|t| >= 3.04** (Sidak, alpha 0.05). Everything else in the
tables is exploratory and is read as such.

```text
HYPOTHESIS H01 - large-move response
Market:      EURUSD, USDJPY, XAUUSD, USTEC, 1-minute bars, dev
Condition:   none beyond the event
Event:       |z_15| >= k, k in {3, 4, 5}; first crossing, 60-minute cooldown per symbol
Prediction:  signed forward return (sign = direction of the 15m move) != 0
Horizon:     5, 15, 30, 60, 120, 240 minutes; PRIMARY k = 4, 60 minutes
Direction:   two-sided. Liquidity-overshoot (Grossman-Miller) predicts reversal
             (negative); slow information diffusion predicts continuation.
Rationale:   a move far outside its own diurnal volatility is either an order
             imbalance a dealer absorbed at a concession (reverts as inventory
             is laid off) or news being priced (continues while it diffuses).
Falsify:     per-day |t| < 3.04 at the primary cell; or real mean inside the
             20-draw date-shifted placebo band; or no monotone dose-response in k.
```

```text
HYPOTHESIS H02 - scheduled x unscheduled (child of H01)
Market:      as H01
Condition:   H01 events split by whether a scheduled US release (qlab.macro_calendar,
             all tiers) falls in [move start - 2 min, event + 1 min]
Prediction:  unscheduled - scheduled < 0 at 60 minutes, k = 4 (PRIMARY: the difference)
Direction:   unscheduled reverts more than scheduled
Rationale:   an unscheduled jump is more likely a liquidity event (inventory
             concession); a scheduled one is information by construction.
Falsify:     difference per-day |t| < 3.04 (Welch on daily sums); or sign flips
             across instruments.
```

```text
HYPOTHESIS H03 - cross-market shock transmission
Market:      every ordered pair of the four symbols (12 pairs), dev
Condition:   source |z_5| >= 4, 30-minute cooldown
Event:       target's shortfall against its beta-implied move over the same 5 minutes,
             s = beta * r_source - r_target, beta = dev OLS of 5m target on 5m source
Prediction:  target's forward return in the direction of sign(s) is positive
Horizon:     1, 5, 15, 30, 60 minutes; PRIMARY 15 minutes, per pair
Direction:   positive (the lagging market catches up)
Rationale:   attention and inventory are market-specific; a shock that lands in
             one market is priced into related ones with a lag.
Falsify:     per-day t < 3.04 at the primary cell; placebo; or the effect lives
             only at 1 minute (a quote-staleness artefact, not tradable).
Secondary:   direction = sign(beta * r_source) (plain transmission, no shortfall).
```

```text
HYPOTHESIS H04 - compression -> expansion direction
Market:      four symbols, dev
Condition:   trailing 60-minute range, in the same z units, in the dev bottom decile
Event:       first 1-minute close outside that range within the next 60 minutes
Prediction:  signed forward return in the breakout direction > 0
Horizon:     15, 60, 240 minutes; PRIMARY 60 minutes
Direction:   continuation, and larger than the same breakout from a mid-decile
             (40th-60th) range - the control
Rationale:   compressed ranges accumulate resting stops on both sides; the first
             side taken triggers a one-directional cascade.
Falsify:     per-day t < 3.04; or compressed - control <= 0; or the mid return
             is positive and the net is not (the usual outcome here).
Observation: forward realised range ratio (expansion itself) is reported
             regardless - volatility is forecastable here, direction usually is not.
```

### Batch 1 results - **24 of 24 primary cells fail**

`python scripts/research/discovery_batch1.py --placebos 20` (dev; Sidak threshold
computed by the script as **3.07**, not the 3.04 estimated above). Mid-price
per-day t at each primary cell:

| hypothesis | cells | best per-day t (mid) | inside placebo band | net > 0 |
| --- | ---: | --- | ---: | ---: |
| H01 large move, k 4, 60 m | 4 | USDJPY +1.36 | 2 of 4 | 1 of 4 |
| H02 unscheduled - scheduled | 4 | USDJPY -1.18 (Welch, daily) | - | - |
| H03 catch-up, 15 m | 12 | EURUSD->USTEC **-2.15** (wrong sign) | 6 of 12 (all six outside are below it) | 0 of 12 |
| H04 compressed breakout, 60 m | 4 | USDJPY **+2.89** | 3 of 4 | 1 of 4 (+0.08) |

- **H01: after a 4-sigma 15-minute move the market is a random walk.** Mean
  max-favourable and max-adverse excursion over the next hour are equal within
  noise on every instrument (USTEC +30.9 / -28.6 bps, gold +22.7 / -23.0).
  There is no dose-response in k: signs change between k = 3, 4 and 5 and
  across horizons, and year by year the sign flips on every instrument. The
  inventory-overshoot story and the diffusion story are both absent.
- **H02: the scheduled/unscheduled split does no work** (|t| <= 1.18). This
  repeats, from the other side, the scheduled-flows finding that fading CPI and
  NFP does not transfer.
- **H03: there is no catch-up.** A target that under-reacts to a shock in a
  related market keeps under-reacting - the shortfall variant is negative in
  9 of 12 pairs at 15 minutes, significantly in none. Nothing at 1 minute
  either, so the four feeds are not stale against each other at this scale.
- **H04: compression predicts expansion, not direction.** The forward 60-minute
  range is **1.45-1.62x** the compressed window's range against 0.99-1.04x for
  the mid-decile control on all four instruments - an OBSERVATION (it is partly
  regression to the mean of a bottom-decile width, and in any case direction-free
  and so not tradable in a spot account). Breakout direction is flat except
  on USDJPY.

**What carries forward.** One thread is coherent across three independent
constructions: **USDJPY continues and the other three do not.** H01 USDJPY is
positive at every k and at 17 of 18 (k x horizon) cells; H04 USDJPY is the only
compressed breakout that continues (+0.42 bps at 15 m, t 3.45; +0.76 at 60 m,
t 2.89; +0.97 at 240 m), and it beats its control; the USDJPY study already
measured minute-scale *momentum* (t +19.8 at 120 m) where EURUSD and USTEC
revert. It was selected by looking, so it is a **child, not a discovery**: H05.
Second, plain transmission (target trades the beta-implied direction of the
source shock) is positive in 9 of 12 pairs at 15 m, none past t 2.7, and below
cost everywhere: H06, an OBSERVATION-level check.

---

## Batch 2 (pre-registered 2026-09-11, after batch 1, before any batch 2 number)

Script: `scripts/research/discovery_batch2.py`.

**Protocol amendment, stated rather than slipped in.** H05 and H06 were chosen
by reading dev, so dev cannot test them. Each gets **one pre-registered cell,
run once on validation**, with the parameters fixed here - not selected from
the dev grid. The dev grid is descriptive and acts only as a gate: if it fails
the gate, validation is not read.

```text
HYPOTHESIS H05 - USDJPY intraday continuation (child of H01, H04)
Market:      USDJPY; EURUSD, XAUUSD, USTEC as falsification (predicted: not positive)
Condition:   decision times on the hour; non-overlapping holds
Event:       |z_W| >= c for the trailing W-minute move; trade in its direction
Horizon:     hold h minutes, exit at the bar-close quote
PRIMARY:     W = 120, c = 2, h = 120 - the centre of the grid, fixed now
Grid (dev):  W in {60,120,240} x c in {1,2,3} x h in {60,120,240}
Gate (dev):  centre cell mid > 0 AND >= 18 of 27 USDJPY cells mid > 0
Test (val):  centre cell per-day t (mid) >= 1.96, outside a 20-draw placebo band,
             and net > 0 -> EXPLOITABLE (validation). Anything less -> rejected.
Rationale:   the carry currency trends intraday: a funding unwind or build is a
             slow, one-sided flow, and Japanese real-money flows are lumpy.
             Interventions (2022, 2024) are the counter-mechanism and are left in.
```

```text
HYPOTHESIS H06 - plain cross-market transmission, pooled (child of H03)
Market:      all 12 ordered pairs; beta estimated on dev only
Event:       source |z_5| >= 4 (30-minute cooldown); target trades sign(beta * r_source)
Horizon:     15 minutes
Test (val):  pooled over the 12 pairs, per-day t (mid) >= 1.96 -> PREDICTIVE.
             Net is expected negative (mid ~ 0.4 bps against ~ 0.55 cost), so
             this cannot reach EXPLOITABLE and is registered as an observation.
```

### Batch 2, H05 dev grid - **gate passed**

`python scripts/research/discovery_batch2.py grid`. Trailing-move continuation,
hourly decisions, non-overlapping holds, 27 cells per instrument, dev:

| instrument | cells mid > 0 | cells net > 0 | centre cell mid / net (t, per day) |
| --- | ---: | ---: | --- |
| **USDJPY** | **27 / 27** | 20 / 27 | **+2.42 / +1.76 bps (t +2.51)** |
| EURUSD | 13 / 27 | 9 / 27 | +0.58 / +0.02 (t +0.69) |
| XAUUSD | 16 / 27 | 7 / 27 | +0.49 / -0.50 (t +0.33) |
| USTEC | 2 / 27 | 2 / 27 | -0.81 / -1.82 (t -0.42) |

USDJPY is a plateau, not a peak: every cell positive, the effect rises with the
threshold `c` at most (W, h) (for example W 240 h 120: +0.98 / +1.46 / +7.22 bps
at c 1 / 2 / 3), and 20 of 27 cells survive cost. No single cell clears a
27-cell Sidak bar (best t +3.20), which is why the test is the fixed centre cell
on validation rather than anything here. The falsification set behaves as the
mechanism requires: USTEC leans the other way (it reverts at these horizons, as
the USTEC study found), and EURUSD and gold are coin flips.

Dev diagnostics on the centre cell, run before validation was read:

- **Not the drift.** Subtracting the unconditional 120-minute return for the
  same hour and year, signed by the trade: +2.45 bps against +2.42 raw.
- **Every dev year positive** (2020 +1.75, 2021 +1.23, 2022 +3.28, 2023 +2.83),
  none individually significant.
- **Outside its dev placebo band**, barely: 20 date-shifted draws span
  [-2.03, +2.00].
- **Asymmetric.** Up-moves continue (+4.03 bps, t 3.13); down-moves barely
  (+0.99, t 0.74). MFE / MAE over the hold +15.9 / -13.5 bps.

### Batch 2 validation, run once - **both fail**

`python scripts/research/discovery_batch2.py validate --placebos 20`. Diurnal
factor and betas from dev; validation supplies only outcomes.

| cell | n | mid bps | net bps | per-day t (mid) | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **H05 USDJPY W120 c2 h120** | 261 | **+1.07** | **+0.25** | **+0.58** | FAIL (needs 1.96); inside placebo [-3.19, +4.12] |
| H05 EURUSD (falsification) | 256 | -0.87 | -1.44 | -1.03 | as predicted |
| H05 XAUUSD (falsification) | 332 | -1.96 | -2.53 | -1.04 | as predicted |
| H05 USTEC (falsification) | 297 | -2.16 | -3.02 | -0.92 | as predicted |
| **H06 pooled, 12 pairs, 15 m** | 8,163 | **-0.06** | -0.82 | **-0.23** | FAIL |

**H05 is rejected under its pre-registered rule, and the rule was badly
powered - which is my error, not the data's.** Validation's per-event standard
deviation is 28.4 bps; at the dev effect of +2.42 bps, 261 events give an
*expected* t of **1.38**. A test that fails at t 1.96 roughly two times in three
even when the effect is exactly as large as dev says cannot settle the question
either way. The power calculation belonged in the registration and was left out.
What the data does say:

- **Pooled dev + validation:** +2.01 bps at the mid (t +2.32), **+1.31 net**
  with a 95% interval of **[-0.33, +2.95]** (t +1.50). Positive in every dev
  year and both validation half-years (2024 +0.37, H1 2025 +2.46), never
  significant in any of them.
- **The dev asymmetry reversed.** Up-moves carried dev (+4.03 vs +0.99 bps); on
  validation it is down-moves (+3.72 vs -1.96), read descriptively after the
  verdict. So the sub-structure is noise and H07 ("up-moves only") is recorded
  and **not** run - validation is spent for this family, and the only clean
  data left is `test`, which H05 did not earn.
- **Instrument specificity held out of sample.** All three falsification
  instruments are negative on validation, as on dev.

Classification: **OBSERVATION.** USDJPY shows 1-4 hour continuation at the mid
in two independent periods, instrument-specific, with a dose-response in move
size on dev - and at ~1.3 bps net against a 0.63 bps round turn it is not
established as exploitable. **H06** is a clean null: plain transmission is zero
on validation.

---

## State of the program after two batches

Seven hypotheses registered, six run, **none survives**. Test split unspent.

**Ranked by evidence** (not by backtest profit):

1. **H05 USDJPY continuation** - OBSERVATION. Positive in both splits at the
   mid, instrument-specific in both, dose-response on dev, pooled net t 1.50.
   The one thread worth more data.
2. **H04 compression predicts expansion** - OBSERVATION, strong and
   unsurprising (forward range 1.45-1.62x vs 1.0x control, all four
   instruments). Direction-free, so a spot account cannot monetise it; it is a
   sizing input, and the USTEC overlay already uses volatility forecasting.
3. Everything else - H01, H02, H03, H06 - is a null at the mid, not a
   cost-killed edge.

**What the failures say about the corpus.** Batch 1 tested whether conditioning
on the tails rescues the minute-scale predictability the earlier studies found
too small. It does not: after a 4-sigma move, max-favourable and max-adverse
excursions are equal, on every instrument. The fixed round turn is not beaten by
waiting for bigger moves, because the conditional *mean* does not grow with the
move - only the variance does.

**Failure conditions worth recording.** A single-cell validation shot on ~260
events cannot confirm a 2-bps effect on a 28-bps-sd outcome. Future
registrations state the expected t at the dev effect size before the shot is
taken, and a cell whose expected t is below ~2.5 is either pooled across a
pre-registered family or not taken.

**Next batch, prioritised (not yet registered):**

1. **USDJPY continuation on ticks, powered by construction** (child of H05).
   The real constraint is sample size, not signal. The honest routes are
   (a) the locked test split, spent as H05's single shot with the power stated
   first - a decision for the project owner, since H05 did not pass the rule
   that earns it; or (b) a pre-registered *family* on validation-free data,
   which this corpus does not have.
2. **Weekend as the one genuinely closed window** - the overnight-reversal
   study's weekly CO-OC row (t +1.07) is the only place that paper's ordering
   survived, and every instrument here shares the same Friday close and Sunday
   reopen. Cross-sectional, so its N is 4; low prior.
3. **Volatility-sized USDJPY exposure conditioned on continuation state** -
   combines the two observations above, but it is a strategy built on two
   non-edges, and the USDJPY overlay already failed on test. Lowest priority.

---

## Batch 3 (pre-registered 2026-09-11, before any batch 3 number)

Script: `scripts/research/discovery_batch3.py`. New information, not new
thresholds: batches 1-2 conditioned on price alone. This batch conditions on
the two other things the feed carries - the **spread** and the **tick count** -
and on the one window in which all four markets are shut together.

Primary cells: 4 (H08) + 4 (H09) + 1 (H10) = **9**; Sidak |t| threshold
computed by the script (about 2.77). Dev only until a cell passes.

**Validation rule, fixed now (the batch 2 lesson).** A dev-passing cell is taken
to validation only if its expected per-day t there - dev mean over dev per-day
sd, times the square root of the validation day count, which is computed from
event *timestamps* only - is **>= 2.5**. Below that the instruments that passed
with the same sign are pooled into one cell and the check repeated; below it
again, the shot is not taken and the cell is reported as underpowered.
Validation pass: per-day t (mid) >= 1.96, same sign, net > 0.

```text
HYPOTHESIS H08 - spread-shock reversal
Market:      four symbols, 1-minute bars
Condition:   bar mean spread >= max(5x trailing-1,440-bar mean, 0.5 bps); rollover
             hours 21-22 UTC, Sundays and the first 60 min after any >= 30-min
             gap excluded; 60-minute cooldown
Event:       entry at the first bar within 15 min whose closing spread is back
             under 2x the trailing mean; D = mid move from 5 min before the shock
             to entry; trade -sign(D)
Horizon:     15, 30, 60, 120 min from entry; PRIMARY 30 min, per symbol
Direction:   positive (reversal)
Rationale:   when dealers withdraw, few quotes set the mid and it overshoots;
             the displacement is paid back once liquidity returns (liquidity-
             provision premium, Nagel 2012). Entry waits for the spread to
             normalise, so the trade pays a normal round turn, not the shock's.
Falsify:     per-day t below threshold; inside the placebo band; or no
             dose-response in |D|.
```

```text
HYPOTHESIS H09 - activity-conditioned reversal (child of H01)
Market:      four symbols, 15-minute decision grid
Event:       |z_15| >= 2; activity A = ticks in the 15 min against the mean of the
             same 15-minute UTC bucket over the prior 20 days; trade the fade
Groups:      dev terciles of log A among events (cut-offs frozen for validation)
Horizon:     15, 30, 60 min; PRIMARY: high-tercile fade minus low-tercile fade at
             30 min, Welch on daily sums, per symbol (two-sided)
Direction:   Campbell-Grossman-Wang (1993): high-activity moves are liquidity
             trades and revert more (difference > 0). Llorente et al. (2002):
             where trading is informed, high-activity moves continue (< 0).
Falsify:     |t| below threshold on the difference.
```

```text
HYPOTHESIS H10 - weekend reopen reversal (from USTEC rejection #15)
Market:      four symbols pooled; one event per symbol per weekend
Event:       reopen = first bar after a >= 24 h gap; entry 60 min after the reopen
             (spreads normalised); G = mid at entry vs Friday's last close;
             trade -sign(G)
Horizon:     exit 4 h, 8 h, 24 h after entry; PRIMARY pooled 8 h
Direction:   positive (reversal)
Rationale:   the reopen prints weekend news through the thinnest book of the
             week; the overshoot is paid back when Asia and London arrive.
             USTEC alone was +9.5 / +5.0 bps (dev / validation) with a hit rate
             under 50%, rejected there on payoff shape rather than significance.
Control:     the same trade demeaned by each symbol's average signed Monday
             return, so a drift in the direction mix cannot pass as reversal.
Falsify:     per-day t below threshold, raw or demeaned.
```

### Batch 3 dev results - **0 of 9 primary cells pass**

`python scripts/research/discovery_batch3.py dev --placebos 20` (Sidak 2.77).

| cell | n | mid bps | net bps | per-day t (mid) |
| --- | ---: | ---: | ---: | ---: |
| H08 EURUSD 30 m | 233 | +0.65 | +0.07 | +0.55 |
| H08 USDJPY 30 m | 140 | -3.24 | -3.97 | -1.66 |
| H08 XAUUSD 30 m | 272 | +0.14 | -0.79 | +0.10 |
| H08 USTEC 30 m | 49 | -14.27 | -15.57 | -1.70 |
| H09 high - low activity, 30 m | 2,244-2,551 | -0.66 to +0.79 | - | -1.13 to +1.02 |
| H10 pooled 8 h | 812 | +0.99 | -0.29 | +0.66 (demeaned +0.81) |

- **H09 is a null.** Activity does not sort the fade in either direction on any
  instrument, so neither Campbell-Grossman-Wang nor Llorente et al. is visible
  through a tick-count proxy.
- **H10 splits by asset class, not by gap.** USTEC and gold reverse the weekend
  (+4.7 / +3.0 bps at 8 h; gold +13.4 at 24 h, t 2.00), both FX majors continue
  it (-1.5 / -2.2). Pooled it is noise, and a two-instrument subset chosen after
  seeing this would be exactly the selection the registry exists to expose.
- **H08 is backwards, and consistently so.** The registered sign was reversion.
  At 120 minutes the fade is negative on **all four** instruments - USDJPY -8.88
  bps (t -2.63), gold -5.72 (t -2.44), USTEC -6.94, EURUSD -2.39 - and the loss
  grows with horizon on every one of them. The move made while the spread was
  blown out *continues* after the spread comes back.

**That last result has a better mechanism than the one registered.** In
Glosten-Milgrom a dealer widens the spread when the flow in front of it looks
informed. A widening is then a label on the move, not noise around it, and an
informed move should keep going as the information diffuses. It also agrees with
H05: the instrument where it is strongest is the one that continues anyway. It
was read off dev, so it is a child and dev cannot test it.

```text
HYPOTHESIS H11 - spread-shock continuation (child of H08)
Market:      four symbols pooled; events exactly as H08 (definition unchanged)
Event:       trade +sign(D) - follow the displacement - from the same entry bar
Horizon:     PRIMARY 120 min, pooled over the four instruments, one cell
Direction:   positive
Rationale:   Glosten-Milgrom adverse selection: spreads widen on informed flow,
             so the displacement made during the widening carries information.
Power rule:  expected validation per-day t = dev per-day mean / dev per-day sd x
             sqrt(validation event days, from timestamps only). Taken only if
             >= 2.5 (registered in batch 3).
Test (val):  per-day t (mid) >= 1.96, positive, and net > 0.
```

### H11 power check - **validation shot not taken**

Dev, pooled over the four instruments, 120 minutes, following the displacement:
**+5.30 bps at the mid, +4.58 net, per-day t +2.67 (net +2.31)** on 671 events
over 353 days. Per instrument: USDJPY +8.88 (t 2.63), gold +5.72 (t 2.44),
USTEC +6.94 (t 0.48, n 39), EURUSD +2.39 (t 1.25).

Validation, counted from event timestamps only - no outcome was read:

| | EURUSD | USDJPY | XAUUSD | USTEC | days with an event |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev events | 226 | 134 | 272 | 39 | 353 |
| **validation events** | **13** | **64** | **3** | **3** | **79** |

Expected validation per-day t = 10.08 / 70.98 x sqrt(79) = **1.26**, against the
registered 2.5. Under the rule the shot is not taken, and validation remains
unread for H11.

The event count is itself the finding, and it is a feed-regime fact, not a
market one. H08 events by quarter (timestamps only) run 20-100 a quarter through
Q3 2023 and then stop on gold, EURUSD and USTEC from Q4 2023. Over the same
quarters the broker's spread distribution was rebuilt: gold's 99.9th-percentile
bar spread fell from ~2.5 bps to 0.23-0.43, and USTEC's *median* fell from ~1.0
bps to 0.014-0.028 in Q4 2024. USDJPY is the exception - its tail spreads
widened and it still fires ~15 times a quarter. Dev's H11 is positive in every
year (2020 +4.08, 2021 +2.76, 2022 +3.26, 2023 +8.10 bps; ex-2020 t +2.60), so it
is not a COVID artefact - but the trigger measures *how this broker quoted in
2021-2023*, and a rule whose event definition depends on a quoting regime that
has since changed is not deployable as written. **Classification: OBSERVATION**
(dev-only, unvalidated, feed-regime-dependent).

---

## State of the program after three batches

Eleven hypotheses registered, nine run to a verdict, **none validated**. Two
validation shots declined or failed on power. Test split unspent.

**The two threads that recur.** Both point the same way - *continuation*, not
reversal, and both are strongest on USDJPY:

| thread | dev | validation | why it stops |
| --- | --- | --- | --- |
| H05 USDJPY 2h continuation | +2.42 mid / +1.76 net, t 2.51 | +1.07 / +0.25, t 0.58 | expected val t was 1.38 |
| H11 spread-shock continuation | +5.30 / +4.58, t 2.67 (USDJPY +8.88) | not read | expected val t 1.26; trigger regime ended in 2023 |

**Why nothing here can be validated, as arithmetic.** The effects this corpus
shows are 1-9 bps a trade against a per-event sd of 28-70 bps - a per-trade
Sharpe of 0.05-0.10. At ~170 trades a year, reaching an expected t of 2.5 needs
roughly (2.5 / 0.07)^2 ~ 1,300 trades, about **7-8 years of independent data**.
Validation is 1.5 years and test 1.2. So a real effect of this size cannot be
confirmed on the held-out data this corpus has, and anything that *does* clear a
single-shot test here is more likely to be the tail of a search than a larger
effect. That is the same power wall the USTEC study hit at daily frequency,
arrived at from the intraday side.

**What would actually move the answer:**

1. **More out-of-sample time on the two continuation threads**, collected
   forward: paper-trade H05 (hourly USDJPY continuation, W 120 c 2 h 120) and
   the USDJPY leg of H11, pre-registered as they stand, for 12-24 months.
2. **Spend the locked test split on a single pooled USDJPY-continuation cell.**
   The expected t is still below 2 even pooled, and neither thread passed the
   rule that earns the split - so this is the owner's decision, not the
   program's.
3. **Stop searching this corpus for intraday direction.** Eleven hypotheses here
   and twenty-five studies before them converge: volatility is forecastable,
   direction at 1-4 hours shows at most a USDJPY-shaped continuation too small
   to confirm on held-out data.

---

## Batch 4 (pre-registered 2026-09-11, before any batch 4 number)

Script: `scripts/research/discovery_batch4.py`. Output: `reports/discovery/batch4/`.

**Why this shot and not another.** The power wall above has a corollary that
decides what is still worth testing. Validation holds about 390 trading days, so
confirming a strategy there at a per-day t of 2.5 needs a daily Sharpe of
2.5 / sqrt(390) = 0.127, which is **about 2.0 annualised, net**. At 1.96 it is
about 1.6. Anything that trades monthly or daily (turn of month, month-end
rebalancing, pre-FOMC, time-series momentum) cannot reach that on this corpus
whatever its true size, so none is registered. Only a strategy that trades many
times a day on every instrument can. Every earlier high-frequency test here was
**linear** (minute AR, cross-asset R-squared) or fitted to the **wrong target**
(the decision-tree paper's next-bar sign, at 113-134 turns a day). The untested
object is a nonlinear forecast of a horizon long enough to clear cost, traded
only when the forecast itself clears cost.

```text
HYPOTHESIS H12 - cost-aware nonlinear forecast (parents: USTEC #8 minute AR,
                 #9 cross-asset lead-lag; the eleven conditioning variables of H01-H11)
Market:      EURUSD, USDJPY, XAUUSD, USTEC; one model per target instrument
Features:    known at the close of minute t, 24 per target:
             own  - z-scored log move over 1, 5, 15, 60, 240 min (batch-1 z);
                    60m/1440m volatility ratio; sigma relative to its 20-day mean;
                    spread relative to trailing 1,440-bar mean; tick count
                    relative to trailing mean; return since 00:00 UTC (z);
                    position in today's range; distance to yesterday's high and
                    low (sigma units); diurnal factor
             cross- z5, z15, z60 of each of the other three instruments
             time - 15-minute UTC bucket, weekday
Target:      30-minute forward mid log return, bps, winsorised at the dev-train
             0.5/99.5 percentiles; contiguous windows only
Model:       sklearn HistGradientBoostingRegressor, fixed a priori: 200 iterations,
             learning rate 0.03, 31 leaves, min 5,000 samples per leaf, l2 1.0,
             no early stopping (sklearn's split is random, which leaks here)
Fit:         dev-train = dev before 2023-01-01, rows on a 3-minute grid
Inner hold:  dev-holdout = calendar 2023 (inside dev). It chooses the entry
             threshold theta = m x the instrument's dev round turn (bps),
             m in {1.0, 1.5, 2.0}, by pooled per-day net t. Nothing else is tuned.
Refit:       same settings on all of dev; theta multiplier kept
Trade:       every minute, if |forecast| >= theta, enter sign(forecast) at the
             bar-close quote, hold 30 min, one position per instrument at a time;
             bid/ask fills + commission + measured slippage of the traded split
PRIMARY:     pooled over the four instruments, per-day t of NET bps, validation
Gate (dev):  dev-holdout pooled per-day net t >= 2.05. Expected validation t is
             then >= 2.05 x sqrt(390 / 250) = 2.56, which is the batch-3 power
             rule. Below the gate, validation is not read.
Pass (val):  per-day net t >= 1.96 and net mean > 0, read once. A pass earns the
             test split (single shot, power stated first); a fail rejects H12.
Controls:    (1) the same trade times with the direction randomised - measures
             what the timing alone costs; (2) a ridge regression on the same
             features - what nonlinearity adds; (3) forecast-decile table of the
             mid return on dev-holdout - predictive power with cost set aside.
Falsify:     gate missed; or holdout decile table not monotone; or validation fail.
Rationale:   linear minute effects here are 0.2-0.5x a round turn *on average*.
             If they are concentrated in states defined by interactions (volatility
             regime x move x time of day x cross-market state), a tree can find
             the subset where the conditional mean clears cost, and a cost
             threshold trades only that subset. If the structure is additive and
             small everywhere, the model finds nothing and that closes the
             question for this feature set.
```

Multiplicity: one primary cell; theta is chosen among three values on dev-holdout,
which is inside dev, so validation is read with no choice left to make.

### Batch 4, H12 dev-holdout - **gate failed; validation not read**

`python scripts/research/discovery_batch4.py dev` (output
`reports/discovery/batch4/`). Models fitted on 288k-335k rows per instrument
(dev before 2023), read on calendar 2023.

| threshold | trades | days | mid bps | net bps | per-day t (mid) | per-day t (net) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 x round turn | 22,444 | 308 | +0.26 | -0.69 | +2.63 | **-6.88** |
| 1.5 x | 11,912 | 299 | +0.26 | -0.75 | +1.59 | -4.49 |
| 2.0 x (chosen) | 6,167 | 288 | +0.37 | -0.72 | +1.44 | **-2.84** |

- **The model has no skill out of sample.** Correlation of the forecast with the
  realised 30-minute return on a non-overlapping grid: EURUSD -0.000, USDJPY
  +0.024, XAUUSD +0.016, USTEC -0.004 (ridge: +0.002, +0.014, -0.010, -0.025).
  The forecast-decile tables are not monotone on any instrument, and the forecasts
  spread 1.7-5.1 bps from bottom to top decile against 0.2-1.1 bps realised - the
  trees fitted dev noise about five times over.
- **The controls agree.** Randomising the direction at the same trade times gives
  a mid of +0.25 / +0.28 / -0.09 / +0.10 bps at 1.0x against the model's +0.08 /
  +0.40 / +0.26 / +0.29, so the forecast is worth 0.1-0.2 bps where it is worth
  anything. The linear model is no better.
- **The result is a bound, not a tuning problem.** The most a flexible model on 24
  state variables extracts at 30 minutes is about +0.3 bps at the mid. This
  account's *commission alone* is 0.2-0.45 bps a round turn, before any spread or
  slippage. No execution style - passive limits included, which on a retail CFD
  fill at the touched quote and so save only the slippage term - turns +0.3 bps
  into a profit here.

Classification: **rejected**. Batch-1 through 4 together now say the same thing
from conditioning (H01-H11) and from flexible fitting (H12): at 30 minutes, the
forecastable mean is below this account's cost floor on all four instruments.

```text
HYPOTHESIS H13 - H12 at 120 minutes (child of H12; the horizon of H05 and H11)
Change:      HORIZON 30 -> 120 for the target, the hold and the cooldown.
             Everything else - features, model settings, theta list, split scheme,
             gate (dev-holdout pooled per-day net t >= 2.05), pass rule - unchanged.
Rationale:   the only effects of 1-9 bps seen in this program live at 1-4 hours
             (H05 continuation, H11 spread-shock continuation) and differ in sign
             by instrument, which a per-instrument model can represent and a
             pooled single rule could not. At 120 minutes a pooled net edge near
             1 bps on ~4 trades a day per instrument would give a daily Sharpe
             near 0.13 - the level validation can confirm. This is the last
             horizon for this feature set; a fail closes it.
Multiplicity: the second draw from the H12 well, stated as such. Validation for
             H13 is read only through the same gate.
```

### Batch 4, H13 dev-holdout - **gate failed; validation not read**

`python scripts/research/discovery_batch4.py dev --horizon 120` (output
`reports/discovery/batch4_h120/`).

| threshold | trades | mid bps | net bps | per-day t (mid) | per-day t (net) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 x round turn | 9,619 | +0.33 | -0.55 | +1.46 | -2.42 |
| 1.5 x (chosen) | 9,243 | +0.36 | **-0.52** | +1.44 | **-2.06** |
| 2.0 x | 8,685 | +0.19 | -0.70 | +0.70 | -2.51 |

Per instrument at 1.0x, mid / net: EURUSD -0.35 / -0.92, USDJPY +0.56 / -0.08,
XAUUSD +1.28 / +0.48 (net t +1.02), USTEC -0.29 / -1.71. The forecasts are again
over-dispersed - their sd is 2.0-5.5 bps against realised outcomes that do not
separate - so the threshold hardly changes the trade count (9,619 at 1.0x,
8,685 at 2.0x): the model trades almost always, and on noise. At 2.0x the
random-direction control beats the model on USTEC (+0.59 vs +0.18 mid). Gold's
positive net is one instrument of four, not significant, and chosen by reading
this table; it is recorded and not pursued.

Classification: **rejected**. This closes the feature set at 30 and 120 minutes.

---

## State of the program after four batches

Thirteen hypotheses registered, eleven run to a verdict, **none validated**.
Validation read twice (H05, H06), declined twice on power (H11, and H12/H13 at
their gates). Test split unspent.

**What four batches establish, as one statement.** On this feed and this account
type, the forecastable part of a 30-minute to 2-hour return is **about 0.2-0.4
bps at the mid**, whether it is found by conditioning on the tails (H01-H11) or
by a flexible model over 24 state variables (H12-H13). The round turn is 0.54-0.72
bps now and was 0.59-1.21 in dev, and commission alone - which no execution
style avoids - is 0.2-0.45 bps. So the gap is not closed by a better signal of
this kind, a better entry, or a cheaper era.

**What this leaves, and none of it is a search the program can run on its own:**

1. **More history, not more hypotheses.** Daily and monthly premia with published
   mechanisms (time-series momentum, pre-FOMC drift - already +33 bps a meeting
   in both splits here at t 1.42 on 43 meetings - month-end rebalancing, turn of
   the month) are untestable on six years because they cannot reach a daily Sharpe
   of 0.13 on 390 days. Twenty-plus years of daily closes for the four
   instruments would test them properly. That needs an external data download.
2. **Forward time** for the two USDJPY continuation observations (H05, H11):
   paper-trade them as registered for 12-24 months.
3. **Different information.** Everything here is built from bid, ask and a
   timestamp. Traded volume, order-book depth, positioning (COT) or options
   data are what the literature's intraday edges are built on, and the feed
   carries none of them.

The deployable result in this repository remains the USTEC risk-managed long.

---

## Batch 5 (pre-registered 2026-09-11, before any batch 5 number)

Script: `scripts/research/discovery_batch5.py`. Output: `reports/discovery/batch5/`.

**Why this shot.** Every earlier test here was one fixed cell (H01-H11) or an
average over all states (minute AR, H12-H13). Neither can see an edge that
lives only in a narrow conjunction - *this* level, taken by *this* much, given
back within *this* many minutes, in *this* session and volatility state. A
grid over those conjunctions is thousands of cells, so the multiplicity is the
whole design problem, and it is solved here by three devices fixed in advance:
a split of dev into two interleaved halves, false-discovery control on the
first, and the entire screen re-run on placebo outcomes to count how many
survivors chance alone produces. Many small edges also cannot be validated one
at a time (the power wall above), so survivors are validated **as one
portfolio**, which is the only unit the held-out data can confirm.

```text
HYPOTHESIS H14 - conditional micro-structure screen (a family, one registration)
Parents:     H01 (extreme move), H03 (cross-market), H04 (range), USTEC #8
             (minute AR), level-interaction + osler (sweeps), orb15 (opening)
Market:      EURUSD, USDJPY, XAUUSD, USTEC; 1-minute bars; dev only
Halves:      A = odd calendar months of dev (discovery), B = even months
             (confirmation). Interleaved so both halves span every regime,
             including the Q4 2023 spread-regime break.
Exclusions:  entries 16:30-18:30 New York (rollover / halt); the Sunday reopen
             and the first 60 minutes after any gap >= 30 minutes; a forward
             window that is not exactly h contiguous minutes is dropped.

Event families (entry at the event bar's close; direction as stated):
  SW  sweep-reclaim  a breach episode starts at the first bar whose high (low)
                     exceeds level L while the prior bar did not; L = prior N-
                     minute extreme, N in {30, 60, 240}, or the prior trading
                     day's (17:00 NY boundary) high/low (PD). If a close is back
                     inside L within K bars (K in {1,3,5}, the breach bar
                     included), the event is that bar; direction = against the
                     breach. Depth = max excursion beyond L / ATR30, bucketed
                     [0,0.15), [0.15,0.40), [0.40,1.0]; deeper excluded.
                     ATR30 = mean range of the prior 48 completed 30-min bars.
  BO  breakout-hold  the same episodes with no reclaim within K bars; event at
                     bar K; direction = with the breach; same depth buckets.
  XM  extreme move   |z_W| >= k, W in {5,15}, k in {2.5,3.5} (batch-1 z);
                     first crossing, 15-min cooldown; direction = fade.
  AR  conditional AR every 5th minute where |z_W| >= 1, W in {5,15};
                     direction = -sign(r_W) (reversal).
  OD  opening drive  anchors 09:00 Tokyo, 08:00 London, 09:30 New York (local,
                     DST-aware); at anchor + M, M in {5,15,30}; direction =
                     sign of the move since the anchor (continuation).
  DX  dislocation    EURUSD, USDJPY, XAUUSD: e = r15 - beta * r15 of the other
                     instrument(s)' USD leg (beta: OLS on half A), scored by its
                     trailing 1,440-bar sd; |e/sd| >= 2, 15-min cooldown;
                     direction = -sign(e) (the dislocation closes).
  Both sides of every base are cells: "side -1" trades the opposite direction.

Conditions (point-in-time, categorical); cell = base x condition x horizon x side:
  TW  New York local: Asia 18:30-02:00, London 02:00-08:00, NY-AM 08:00-11:30,
      NY-PM 11:30-16:30
  VR  sigma (trailing 1,440-bar, diurnal-adjusted) / its trailing 20-day mean:
      < 0.8, 0.8-1.25, > 1.25
  USD USD move of the *other* instruments over 30 min (z), read as the direction
      it implies for the target (USD up => EURUSD, XAUUSD, USTEC down; USDJPY
      up), against the trade: aligned (z > 0.5), opposed (z < -0.5), flat
  TR  trading-day-to-date move (z) against the trade: with (> 1), against
      (< -1), flat
  Set: none, each level of TW/VR/USD/TR alone (13), TW x VR (12), TW x USD (12)
      = 38 conditions
Horizons:    1, 3, 5, 10, 20, 60 minutes

Outcome:     net_now = signed mid return (bps) - RT(symbol, UTC hour), where RT
             is the round turn *measured on the validation split* (spread +
             commission + 2 x slippage; adverse fraction 0.5 for fade families,
             1.0 for BO/OD/continuation sides) - the cost a live account pays
             now, not the 2020-2023 cost. Dev-era fill net is reported beside it.
Statistic:   per-day t of net_now (events summed within a day, tested across
             event days), one-sided.
Eligible:    n_A >= 100 events on >= 60 distinct days.
Stage A:     Benjamini-Hochberg over all eligible cells, q = 0.10, net_now > 0.
Stage B:     the same cell on half B: net_now mean > 0 and per-day t >= 1.96.
Placebo:     Stages A+B re-run 3 times with outcomes moved to the same clock
             minute 1-5 trading days away (features and directions kept). If
             the mean placebo survivor count is >= half the real count, the
             screen is declared uninformative, whatever the survivors show.
Stage C:     de-duplicate survivors (greedy by t_A; drop a cell if >= 50% of its
             events fall within 5 minutes of a kept cell's on the same symbol),
             then one portfolio = every kept cell's trades at 1 unit each.
             Expected validation t = portfolio per-day t on B x sqrt(validation
             event days / B event days). Shot taken only if >= 2.5.
Pass (val):  portfolio per-day net t >= 1.96 at real validation bid/ask fills +
             commission + slippage, mean > 0, read once -> ROBUST candidate,
             which earns the test split (single shot, power stated first).
Falsify:     no Stage B survivors; placebo rule; expected-t rule; validation.
Rationale:   if liquidity-provision and stop-cascade effects exist at the
             minute scale they are conditional - on where resting orders sit,
             how deep the probe went, how fast it was rejected, and who is
             trading - and an unconditional average dilutes them below cost.
```

Multiplicity: one registration for the whole grid. The false-discovery rate is
controlled on A; B is an independent confirmation; the placebo screen measures
the combined procedure's false-survivor count directly instead of assuming it.

### Batch 5 results - **H14 rejected: 0 of 81,040 cells pass stage A**

`python scripts/research/discovery_batch5.py dev --placebos 3`. 351 base event
series, 2.3M events, 81,040 eligible cells. The best stage-A net t is 3.53,
against the ~4.5 that Benjamini-Hochberg at q 0.10 needs across 81k cells, so
stage A is empty, and so are all three placebo screens. The top 40 cells by
stage-A t go nowhere: 34 of them are negative on half B.

**Diagnostic, read after the verdict (exploratory):**
`scripts/research/discovery_batch5_diag.py` recomputes every cell cost-free and
asks whether cells that are strong in half A are strong in half B.

| | real | placebo 1 | placebo 2 |
| --- | ---: | ---: | ---: |
| cells with \|t_A\| >= 2.5 and \|t_B\| >= 2, same sign | **432** | 22 | 14 |

- **At the mid there is real, replicating conditional structure, 20-30x chance.**
  Split by instrument and family, the correlation of cell t between halves runs
  up to 0.57-0.59 (AR on EURUSD and gold), against 0.2 or less on placebo.
- **It is reversal, nearly everywhere.** AR cells are 100% reversal (EURUSD
  114 of 114, gold 86 of 86); breakouts that hold *reverse* (BO negative in 45
  of 47 EURUSD cells and 25 of 25 gold cells); gold extreme moves fade; EURUSD
  dislocations against the USD leg close. This is liquidity provision priced in
  minutes, and it is the same sign as the linear minute AR the earlier studies
  measured.
- **It is mostly smaller than the round turn**, which is why H14's net screen
  saw nothing: the median replicating cell earns 0.1-0.9x of a validation-era
  round turn. A minority clears it, e.g. EURUSD DX fade against the day's trend
  at 10 minutes, +0.73 / +0.85 bps at the mid (t 4.12 / 5.14) against 0.56 bps.

That minority is the only thing in five batches with both replication across
independent halves *and* a per-trade edge at or above cost. It was selected by
reading both halves of dev, so it is a child and only validation can test it.

---

## Batch 6 (pre-registered 2026-09-11, before any batch 6 number)

Script: `scripts/research/discovery_batch6.py`. Output: `reports/discovery/batch6/`.

```text
HYPOTHESIS H15 - minute-scale liquidity-provision portfolio (child of H14 diagnostic)
Universe:    the side +1 cells of `reports/discovery/batch5/diag_mid_cells.parquet`
             (the H14 grid, cost-free, both halves of dev).
Selection:   |t_A| >= 2.5 and |t_B| >= 2.0 with the same sign (per-day t, mid);
             traded direction = that sign; pooled dev mid edge in the traded
             direction >= 1.0 x the round turn charged to the *traded* side
             (validation-era, adverse fraction 0.5 for a fade, 1.0 for a chase -
             the diagnostic table used the side +1 cost and is corrected here).
De-dup:      greedy by min(|t_A|, |t_B|); drop a cell if >= 50% of its events lie
             within 5 min of a kept cell's on the same symbol.
Portfolio:   every kept cell's trades, one unit each, market entry at the event
             bar's closing quote, exit at the cell's horizon. Positions may
             overlap; P&L summed per trading day.
Power:       expected validation t = dev net_now per-day t on the *weaker* half x
             sqrt(validation event days / that half's event days), event days
             counted from validation timestamps only. Shot taken only if >= 2.5.
Pass (val):  per-day t >= 1.96 at real validation bid/ask fills + commission +
             slippage, mean > 0, read once. Pass -> test split, single shot,
             power stated first. Fail -> H15 rejected.
Placebo:     the same selection run on the batch-5 placebo-1 cells is reported
             beside it, so the share of the portfolio chance would supply is
             stated.
Rationale:   the H14 diagnostic shows reversal structure that replicates across
             interleaved halves at 20-30x the placebo rate. One cell of it is
             unconfirmable on 390 days (the power wall); a portfolio of every
             cell whose edge clears cost trades many times a day on several
             instruments, which is the only shape validation can confirm.
```

### Batch 6, H15 dev - **power gate failed; validation not read**

`python scripts/research/discovery_batch6.py dev`, then `... validate` (which
stops before reading any outcome when the gate fails). 33 cells replicate and
clear the cost of the side they trade; 23 remain after de-duplication - 12 on
gold, 9 on EURUSD, 1 each on USTEC and USDJPY, and 22 of the 23 are fades or
failed-breakout trades. **Placebo 1 supplies 9 such cells by chance against 33
real**, so roughly a quarter of the portfolio should be expected to be noise.

| dev portfolio, net_now | trades | days | net bps/trade | bps/day | per-day t | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| half A | 17,350 | 584 | +0.49 | +14.6 | +3.91 | 2.57 |
| **half B (weaker)** | 16,746 | 596 | **+0.24** | +6.7 | **+2.03** | **1.32** |
| all dev (selected on both halves) | 34,096 | 1,019 | +0.37 | +12.3 | +4.22 | 2.10 |

At the mid the same trades earn +1.17 bps each. Validation holds 388 event
days (timestamps only), so the expected validation t is 2.03 x sqrt(388 / 596)
= **1.64 < 2.5**. The shot is not taken. At half B's effect size it would pass
about 37% of the time, which would spend the last clean split on something close
to a coin flip.

**Classification: OBSERVATION** - a portfolio of replicating minute-scale fades
that is positive net of today's cost in both halves of dev, and too small per
trade to confirm on 1.5 years.

```text
HYPOTHESIS H16 - H15 executed on ticks (child of H15; execution, not signal)
Change:      the assumed slippage term (0.5-1.0 x mean 250 ms drift, always
             adverse) is replaced by the drift *these* trades actually meet.
             Cells, directions, horizons and conditions unchanged (the 23 in
             reports/discovery/batch6/kept.parquet).
Fill:        order sent at the signal bar's close T; entry at the first tick at
             or after T + 250 ms, exit at the first tick at or after T + h + 250
             ms (either more than 5 s late -> trade dropped). Long buys the ask
             and sells the bid, short the reverse.
Dev unit:    net_tick_now = signed mid move between the two fill ticks - the
             validation-era mean spread for the entry hour - commission. (Dev
             spreads were up to 2x today's, so dev fills are reported beside it
             but are not the gate.) Stress latencies 500 and 1,000 ms reported.
Power:       expected validation t = half-B per-day t of net_tick_now x
             sqrt(388 / half-B event days). Shot only if >= 2.5.
Pass (val):  real validation tick fills at 250 ms + commission; per-day t >= 1.96
             and mean > 0, read once. Pass -> test split, single shot.
Rationale:   fades buy into selling; if the order flow that made the move keeps
             arriving for a few hundred milliseconds the drift is adverse, and if
             the move is already exhausted at the bar close it is favourable.
             The cost model cannot know which, and for trades whose edge is
             1-2x the round turn it decides the answer.
Multiplicity: the second draw on the H15 portfolio, stated as such. Validation
             stays unread unless the gate passes.
```

**Amendment to the H16 fill rule (2026-09-11, after the first dev run, before
validation).** The registered rule - fill at the first tick at or after T + 250
ms, drop the trade if that tick is more than 5 s late - is a specification error
for a quote feed that prints only on change. It dropped 7.4% of trades (EURUSD
20%, concentrated in London hours 06-11 UTC, when the market is open and the
standing quote is fillable), and the dropped trades were not random: EURUSD's
dropped trades averaged +0.09 bps net_now against +0.39 for those kept, so the
rule flattered the result. The corrected rule fills at the **quote standing at T
+ latency** (the last tick at or before it) and at T + h + latency, and drops a
trade only if that quote is more than 60 s old (a halt or a closed market). The
first run's figures (half-B t 3.84, expected validation t 3.10) are superseded
and recorded only here. The gate is recomputed on the corrected tape, with
everything else unchanged.

### Batch 7, H16 dev (corrected fills) - **gate passed**

`python scripts/research/discovery_batch7.py dev`. Standing-quote fills drop 223
of 34,791 trades (0.6%, halts only). Per trade, bps:

| half B (the gate half) | trades | net/trade | per-day t | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| bar model, assumed slippage (H15) | 16,746 | +0.237 | 2.03 | 1.32 |
| **ticks, 250 ms, today's spread + commission** | 17,035 | **+0.444** | **3.85** | 2.50 |
| ticks, 1,000 ms | 17,036 | +0.440 | 3.81 | 2.47 |
| ticks, 250 ms, *dev-era* real fills | 17,035 | +0.200 | 1.73 | 1.12 |

- **The drift these trades meet is nil.** The tick mid at 0, 250, 500 and 1,000
  ms is the same to 0.01 bps, so the 0.2 bps a trade that the cost model charged
  as slippage (a fraction of the *absolute* 250 ms drift, always adverse) is not
  paid by a fade entered at a bar close. Latency up to a second does not matter.
- **The edge depends on today's spreads.** At dev-era fills it halves to +0.20
  bps. Validation's real fills are therefore the test of whether the spread
  compression carries the edge with it or the edge was the compensation for the
  wider spread.

Expected validation t = 3.85 x sqrt(388 / 597) = **3.10 >= 2.5**: the single
validation shot is taken, with real validation tick fills at 250 ms + commission,
pass at per-day t >= 1.96 and mean > 0.

### Batch 7, H16 validation, run once - **FAIL, significantly the wrong way**

`python scripts/research/discovery_batch7.py validate`.

| validation, real tick fills | trades | days | net bps/trade | per-day t | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| **250 ms (registered)** | 12,540 | 388 | **-0.437** | **-3.57** | -2.87 |
| 1,000 ms | 12,541 | 388 | -0.433 | -3.55 | -2.86 |

Checked against a code error before it was read as a result: the same validation
events recomputed with the bar model, no tick code involved, earn **+0.09 bps at
the mid (per-day t 0.71)** against +1.17 on dev. The signal collapsed by about
93%, and it did so throughout: every quarter from 2024Q1 to 2025Q2 lies between
-0.17 and +0.32 bps at the mid. 21 of the 23 cells lose after cost; the two that
do not are the smallest (86 and 219 trades) and are recorded, not pursued. The
net loss is the round turn charged on a mid of zero.

**Why the design did not catch it.** The halves were interleaved on purpose, so
that both would span every regime - and that is exactly why they could not
detect a regime ending. A and B confirmed that the reversal structure
*replicates within* 2020-2023; neither could say whether it *persists past* it.
The one warning was on the table before the shot: at dev-era fills the edge
halved (+0.20 bps), the tell of a liquidity-provision premium that is paid for
the wider spread of its era. The broker's spread distribution was rebuilt in
Q4 2023 (batch 3), and in 2024-2025 fading a minute-scale move earns the mid
back to zero. Classification: **rejected**. The test split is unspent.

---

## State of the program after seven batches

Sixteen hypotheses registered, none validated. Validation has been read for
H05, H06 and H16. Test split unspent.

**What batches 5-7 add.** The grid the program had not run - conditional cells
at 1-60 minutes, 81,040 of them - finds **nothing net of today's cost** under
false-discovery control (H14), finds **real reversal structure at the mid** that
replicates across halves at 20-30x the placebo rate (diagnostic), and that
structure **belongs to 2020-2023**: the portfolio built from it, which passed
every dev check and its power gate, loses 0.44 bps a trade in 2024-2025 at
t -3.57 (H16).

**What this changes about where to look.** Dev is a different microstructure
era from the one a live account trades in. Every further search on dev - however
well controlled - measures that era. The only post-regime data are validation
(now read for this family) and test (locked). What remains:

1. **Re-split by regime** - search 2024-H1 2025 and confirm on test, one shot.
   That changes the fixed split protocol and spends test, and test's ~290 days
   confirm only a net daily Sharpe near 2.3 - so it is the owner's decision.
2. **Forward time** in the current regime, paper-traded.
3. **Different information** (volume, depth, positioning), as before.
