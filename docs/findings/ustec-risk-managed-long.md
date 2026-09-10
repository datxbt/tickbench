# USTEC: sixteen rejected edges, and the one thing worth deploying

**Verdict: accepted, with the claim stated narrowly.** The deployed strategy has
**no directional edge and does not claim one.** It harvests the equity risk
premium at a controlled volatility, and what it delivers over a passive long is
the same risk-adjusted return with **roughly half the drawdown**, consistently
across dev, validation and the held-out test split.

Everything else was tried first and failed. That failure is most of this report,
because it is what makes the narrow claim credible.

| | dev 2020-23 | validation 2024-25H1 | test 2025H2-26 |
| --- | ---: | ---: | ---: |
| strategy CAGR | 11.14% | 13.77% | 18.87% |
| strategy Sharpe | **0.95** | 0.91 | 1.15 |
| strategy max drawdown | **-15.1%** | **-12.9%** | **-7.9%** |
| strategy Calmar | **0.74** | **1.07** | **2.40** |
| buy & hold CAGR | 13.15% | 20.01% | 22.81% |
| buy & hold Sharpe | 0.60 | 0.90 | 1.20 |
| buy & hold max drawdown | -39.8% | -24.9% | -10.8% |
| buy & hold Calmar | 0.33 | 0.80 | 2.10 |

Zero swap, costs from the measured tape. Swap is the one term this corpus cannot
measure and it gets [its own section](#the-term-this-corpus-cannot-measure).

Read the table honestly: the strategy **loses to buy and hold on raw return in
all three periods**, and beats it on Sharpe only in dev. What it does in every
period is cut the drawdown - by 62%, 48% and 27% respectively - and that is the
whole product.

- Code: `src/qlab/strategies/risk_managed_long.py`
- Driver: `scripts/backtests/backtest_risk_managed_long.py`
- Expert: `USTEC_RiskManagedLong.mq5` (compiles clean, 0 errors / 0 warnings)
- Raw numbers: `reports/strategies/risk_managed_long.json`
- Note: the dev figures were corrected by 0.05pp after a panel fix made while
  adding the FX session convention - the open is now taken only from a bar that
  actually quoted at 09:30, which drops one dev session that first quoted later.
  Validation and test were unaffected; the test split was re-run once to confirm
  that and returned identical numbers.
- Study page: https://claude.ai/code/artifact/e4202817-fa9b-4031-a163-c8e4767804d9

---

## 1. The rejection log

Sixteen directional hypotheses, each written down before its number was read,
each reported whatever it said. The unit throughout is basis points, against a
USTEC round turn of **1.40 bps** at the cash open - 1.04 spread + 0.63 commission
+ 2 x 0.16 slippage = 1.82 index points at a 13,000 level, on the dev cost
profile at 13:00 UTC. Validation's profile is cheaper at 1.19 bps; every
backtest below is costed with its own split's profile, as `costs.py` enforces.

| # | hypothesis | result | verdict |
| --- | --- | --- | --- |
| 1 | Hour-of-day drift | shape correlation between dev and validation: **-0.11** | noise |
| 2 | Session-halt gap | +5.44 bps, t=4.09 on the retired schedule; **+0.11 bps, t=0.09** on the current one | structurally dead |
| 3 | Intraday momentum, first 30m to last 30m | corr -0.047; net -1.9 bps/trade | no |
| 4 | Open drive, first 30m to rest of session | corr -0.015 | no |
| 5 | Turn of the month | +9.6 vs +3.9 bps, t=1.12 | not significant |
| 6 | Day-of-week, overnight leg | best cell t=2.62 of 5 cells | multiple testing |
| 7 | Overnight reversal on prior day's move | all four buckets t<1.3 | no |
| 8 | Minute-level autoregression | t=**-29** (overlapping), average conditional edge **0.56 bps**; non-overlapping event study: **no gross edge at all**, 36 of 36 cells negative net | too small |
| 9 | Cross-asset minute lead-lag (EUR/JPY/XAU) | R²=0.028%, E\|edge\| **0.047 bps**, 0.01% of minutes clear cost | too small |
| 10 | Cross-asset daily lead-lag | **every sign flipped** dev→validation (EUR -0.069→+0.067, JPY +0.052→-0.070, XAU -0.010→+0.078) | noise |
| 11 | Overnight gap continuation / fade | corr +0.055; fading loses 3.4 bps/day | no |
| 12 | Volatility regime predicting return | low/mid/high buckets +9.4/+10.1/+6.1 bps, all t<1.6 | no |
| 13 | Intraday window selection (dev-select, val-verify) | **dev SR +0.81 → validation SR -1.79** | decisive failure |
| 14 | Daily dip-buying / short-term reversal | broadly positive in dev, mixed in validation, n=26-200 per cell | not established |
| 15 | Weekend gap fade | +9.5 bps dev / +5.0 val, hit rate **below** 50% - a short-volatility payoff shape | rejected on shape |
| 16 | Trend gate as a *return* signal | helps in dev, hurts intraday variants in validation (-7.05% CAGR) | not a return signal |

### The two that are worth dwelling on

**#2, the session-halt gap, is the trap this project was built to catch.** The
gap across USTEC's daily halt averages **+5.44 bps with t=4.09** over dev - a
large, highly significant, apparently free trade. It is an artifact of a broker
schedule that no longer exists.

Reading the bar timestamps month by month shows two distinct regimes, both fixed
in *New York* local time while their UTC hours move with US DST:

| period | halt (New York) | what it spans |
| --- | --- | --- |
| 2020-06 to 2023-05 | 16:00 - 18:30 | the cash close **plus the 16:00-17:00 futures hour** |
| 2023-06 to 2026-09 | 17:00 - 18:00 | exactly the CME maintenance break |

Under the old schedule the CFD was shut while NQ futures kept trading, so the
reopen had to catch up to an hour of real price discovery - a stale-price gap,
and a real one. Since June 2023 the broker's halt coincides with the futures
halt, nothing trades anywhere, and there is nothing to catch up to. The gap
falls to +0.11 bps (t=0.09) on dev's tail and +0.94 bps (t=1.00) over
validation's 287 sessions, against a 1.40 bps cost.

A backtest run over the whole corpus without noticing the schedule change would
have reported a t=4 edge and deployed a strategy into a market that stopped
offering it three years ago.

*(This is also the one place the locked test split was read before the final run:
only bar timestamps, to confirm the current schedule is still the futures one.
No prices, no returns. Knowing an instrument's trading hours is a deployment
fact, not a signal - but it is a read, so it is on the record.)*

**#13, intraday window selection, is the cleanest negative in the set.** Rank
all 46 half-hour New York windows by their dev t-statistic, take every window
with |t| > 1.5 - long five of them, short two - and run that fixed selection
forward:

```
IN-SAMPLE dev  6,133 trades  +3.13 bps/session  t=+1.63  SR +0.81
OUT val        2,578 trades  -5.21 bps/session  t=-2.21  SR -1.79
```

The out-of-sample result is not merely worse, it is significantly negative. That
is what selecting on 46 correlated cells buys.

### Why the whole family was always going to fail

Two arithmetic facts, either of which is sufficient.

**At minute frequency the predictability is real and 20x too small.** The
strongest minute-level relationship in the corpus is 60-minute reversal, at
t=-29 over 1.2M observations. Its beta is -0.028, so its expected edge is
0.028 x the size of the past move - about 0.56 bps on average, against a 1.40
bps round turn. The best multivariate cross-asset model predicts 0.047 bps and
clears the round turn on **0.01% of minutes**.

**At daily frequency there is not enough data to establish anything.** Dev holds
979 sessions with a standard deviation of 131 bps. A strategy with a genuinely
good Sharpe of 0.8 produces t ≈ 1.6 over that window. So dev *cannot* validate a
daily strategy - and anything in it that does reach t > 2 is more likely to be
one of the many cells tried than a real effect. This is not a data-quality
problem, it is a power problem, and no amount of further searching fixes it.

---

## 2. What survived

Direction is not forecastable here. **Risk is.** On non-overlapping 20-session
blocks:

| | corr(vol, next vol) | corr(return, next return) | vol range |
| --- | ---: | ---: | --- |
| dev | **+0.449** | -0.046 | 10.0% - 94.7% |
| validation | **+0.225** | -0.150 | 12.4% - 56.6% |
| test | **+0.151** | +0.084 | 6.1% - 33.4% |

Volatility autocorrelation is positive in all three periods; return
autocorrelation is not distinguishable from zero in any of them. Note the honest
part: **the volatility signal is weaker in the two recent periods than in dev**,
and this is measured on only 14-18 blocks out of sample. What did not weaken is
the thing the strategy actually delivers - the drawdown reduction is present in
all three.

And the range column is the point. A passive long on USTEC is a position whose
risk varies by a factor of six to nine. Holding a constant *notional* means
holding a wildly varying *risk*, and on a leveraged CFD account that is how
accounts die.

---

## 3. The strategy

One decision per US cash session, entirely from closed data.

```
gate    = 1 if cash_close > SMA(cash_close, 200) else 0
vol     = stdev(last 20 cash close-to-close log returns) * sqrt(252)
weight  = gate * clip(15% / vol, 0, 2.0)
trade   = only when |weight - weight_held| > 0.10
```

Six parameters, none of them fitted here: 200 and 20 are the conventional
lengths, 15% is a risk-appetite choice (and §5 shows it is pure leverage), 2.0
is a margin decision, 0.10 is a cost control.

**The timing convention is the part that matters.** The signal is computed from
session *t*'s cash close; the trade happens at session *t+1*'s cash open; the
return the weight earns is measured **open to open**. A weight is therefore never
applied to a return any part of which was observable when the weight was chosen.
This is the one place a sizing strategy leaks, and it leaks silently - nothing
raises, the curve just gets better - so `tests/test_risk_managed_long.py` asserts
the alignment directly rather than trusting it.

### The candidate set

Five structural variants, fixed before evaluation. Selection was made on
validation, which is that split's stated purpose.

**dev**

| strategy | CAGR% | vol% | SR | maxDD% | Calmar | turn/yr | avg w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy and hold | 13.15 | 26.48 | 0.60 | -39.8 | 0.33 | 0.0 | 1.00 |
| A vol-target only | 15.07 | 16.18 | 0.95 | -24.5 | 0.61 | 4.0 | 0.72 |
| B trend gate only | 10.88 | 13.34 | 0.84 | -22.0 | 0.50 | 3.4 | 0.54 |
| **C gate + vol-target** | 11.14 | 11.90 | 0.95 | -15.1 | 0.74 | 4.7 | 0.50 |
| D gate half + vol-target | 13.61 | 13.16 | 1.04 | -18.5 | 0.73 | 4.0 | 0.61 |
| E C, no band | 10.37 | 11.92 | 0.89 | -16.2 | 0.64 | 6.6 | 0.50 |

**validation**

| strategy | CAGR% | vol% | SR | maxDD% | Calmar | turn/yr | avg w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy and hold | 20.01 | 23.41 | 0.90 | -24.9 | 0.80 | 0.0 | 1.00 |
| A vol-target only | 12.81 | 17.00 | 0.79 | -19.2 | 0.67 | 5.5 | 0.84 |
| B trend gate only | 16.12 | 17.62 | 0.94 | -16.0 | 1.01 | 2.7 | 0.89 |
| **C gate + vol-target** | 13.77 | 15.42 | 0.91 | -12.9 | 1.07 | 6.6 | 0.79 |
| D gate half + vol-target | 13.97 | 15.88 | 0.90 | -15.0 | 0.93 | 5.9 | 0.82 |
| E C, no band | 14.50 | 15.24 | 0.96 | -12.2 | 1.18 | 10.0 | 0.79 |

C was chosen: best or near-best Calmar in both splits, the lowest drawdown among
the variants that also beat buy and hold in dev, and the lowest turnover of the
close contenders. It is deliberately **not** the maximum of anything - E edges it
on validation Calmar (1.18 vs 1.07) but loses in dev (0.64 vs 0.74) and trades
50% more; picking the winner of a two-cell coin flip is how the rejection log
above got so long.

### The held-out test split, run once

| strategy | CAGR% | vol% | SR | maxDD% | Calmar | turn/yr | avg w |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy and hold | 22.81 | 18.50 | 1.20 | -10.8 | 2.10 | 0.0 | 1.00 |
| A vol-target only | 19.59 | 16.72 | 1.15 | -10.6 | 1.85 | 5.9 | 0.95 |
| B trend gate only | 20.20 | 17.71 | 1.13 | -10.1 | 2.00 | 1.8 | 0.96 |
| **C gate + vol-target** | 18.87 | 16.25 | 1.15 | **-7.9** | **2.40** | 7.0 | 0.92 |
| D gate half + vol-target | 19.32 | 16.36 | 1.16 | -9.0 | 2.15 | 6.4 | 0.93 |
| E C, no band | 19.17 | 16.16 | 1.17 | -7.7 | 2.48 | 10.0 | 0.91 |

The test period is a strong bull market with no meaningful drawdown - the single
worst environment for a defensive overlay, and it still improved Calmar from 2.10
to 2.40 while giving up 3.9 points of CAGR. The mean session return is +7.39 bps
at t=+1.22, which is not significant and was never going to be over 285 sessions.

**The test split is now spent for this strategy family.**

### Year by year

| year | strategy | buy & hold | avg weight |
| --- | ---: | ---: | ---: |
| 2020 | +6.71% | +34.27% | 0.11 |
| 2021 | +19.90% | +24.68% | 1.02 |
| 2022 | **-10.53%** | **-36.09%** | 0.04 |
| 2023 | +31.38% | +50.77% | 0.79 |
| 2024 | +22.38% | +24.65% | 0.93 |
| 2025 (val part) | -1.41% | +4.64% | 0.51 |
| 2025 (test part) | +14.19% | +12.09% | 1.17 |
| 2026 | +6.48% | +12.55% | 0.73 |

This is the whole character of the strategy in one table. It gives up most of
2020's rebound (it was sized at 0.11 through the COVID volatility) and half of
2026, and in exchange it loses 10.5% in 2022 where a passive long loses 36.1%.
If you want the 2020 number you do not want this strategy.

---

## 4. Costs, and why they barely matter

Turnover is 4.7-7.0x notional a year, which at 1.40 bps a round turn is 3-5 bps
of annual drag. Running the full Stage 6 stress - **3x spread and every fill
fully adverse** - moves the answer by almost nothing:

| | base CAGR | stressed CAGR | difference |
| --- | ---: | ---: | ---: |
| dev | 11.14% | 11.10% | -0.04 pp |
| validation | 13.77% | 13.74% | -0.03 pp |
| test | 18.87% | 18.85% | -0.02 pp |

That insensitivity is the direct payoff of the no-trade band, and it is worth
protecting. A well-meaning change that rebalances daily instead would raise
turnover roughly an order of magnitude and start to matter.

Note the contrast with the rejected hypotheses, every one of which died *because*
of cost. The difference is not that costs got cheaper; it is that this strategy
trades 5 times a year instead of 250.

---

## 5. Parameter sensitivity

A strategy that only works at one setting is a fit. Sharpe over the whole
neighbourhood, both splits:

| parameter | dev SR | dev Calmar | val SR | val Calmar |
| --- | ---: | ---: | ---: | ---: |
| ma_days = 100 | 0.79 | 0.61 | 0.54 | 0.49 |
| ma_days = 150 | 0.92 | 1.15 | 0.90 | 1.05 |
| **ma_days = 200** | 0.94 | 0.74 | 0.91 | 1.07 |
| ma_days = 250 | 0.62 | 0.32 | 0.84 | 0.89 |
| ma_days = 300 | 0.78 | 0.61 | 0.40 | 0.31 |
| vol_days = 10 | 0.59 | 0.42 | 0.82 | 1.03 |
| vol_days = 15 | 0.85 | 0.62 | 0.89 | 1.10 |
| **vol_days = 20** | 0.94 | 0.74 | 0.91 | 1.07 |
| vol_days = 30 | 0.83 | 0.60 | 0.91 | 0.94 |
| vol_days = 60 | 0.79 | 0.49 | 0.72 | 0.70 |
| target_vol = 10% | 0.92 | 0.64 | 0.87 | 1.01 |
| **target_vol = 15%** | 0.94 | 0.74 | 0.91 | 1.07 |
| target_vol = 20% | 0.90 | 0.67 | 0.94 | 1.14 |
| target_vol = 25% | 0.89 | 0.65 | 0.92 | 1.11 |
| max_weight = 1.0 | 0.92 | 0.70 | 0.90 | 1.02 |
| **max_weight = 2.0** | 0.94 | 0.74 | 0.91 | 1.07 |
| max_weight = 3.0 | 0.94 | 0.74 | 0.91 | 1.07 |
| band = 0.0 | 0.88 | 0.64 | 0.96 | 1.18 |
| **band = 0.10** | 0.94 | 0.74 | 0.91 | 1.07 |
| band = 0.20 | 0.96 | 0.70 | 1.08 | 1.23 |
| band = 0.30 | 0.79 | 0.56 | 0.98 | 1.12 |
| **gate_floor = 0.0** | 0.94 | 0.74 | 0.91 | 1.07 |
| gate_floor = 0.25 | 1.03 | 0.79 | 0.95 | 1.09 |
| gate_floor = 0.5 | 1.03 | 0.73 | 0.90 | 0.93 |
| gate_floor = 1.0 (no gate) | 0.94 | 0.61 | 0.79 | 0.67 |

Three things this says:

- **Sharpe is positive at every one of the 27 settings, in both splits.** The
  range is 0.40 to 1.08. There is no cliff.
- **`target_vol` is pure leverage**, as it must be: Sharpe is flat from 10% to
  25% while drawdown scales with it. Choosing 15% is a statement about how much
  risk you want, not a backtest result. `tests/` asserts this algebraically.
- **`ma_days` is the one genuinely sensitive dimension** - 250 dips to 0.62 in
  dev. 200 is the conventional value and was not picked off this table, but the
  dip is real and is the honest weak point of the design.

---

## 6. The term this corpus cannot measure

A tick feed carries quotes, not financing. **Overnight swap is not in any number
above**, and it is the largest single uncertainty in the strategy.

Break-even - the financing rate at which the entire return goes to the broker:

| swap (bps/night) | annual drag, dev | dev CAGR left | val CAGR left | test CAGR left |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00% | 11.14% | 13.77% | 18.87% |
| 0.50 | 0.92% | 10.13% | 12.11% | 16.84% |
| 1.00 | 1.83% | 9.12% | 10.47% | 14.84% |
| 1.50 | 2.75% | 8.13% | 8.86% | 12.87% |
| 2.00 | 3.67% | 7.14% | 7.27% | 10.94% |
| **break-even** | | **6.12** | **4.79** | **5.39** |

A typical index-CFD financing charge of 1.5-2.0 bps/night takes 3-7 percentage
points a year off. That is survivable but it is not nothing, and it must be read
off the live account rather than assumed - `InpSwapCheckBps` in the expert
refuses to trade above a stated limit.

The comparison with buy and hold survives it, because **buy and hold pays the
same swap on a larger average position** (1.00 against 0.50-0.92 here). At 1.5
bps/night, charged to both:

| | strategy | buy & hold |
| --- | ---: | ---: |
| dev CAGR / SR / maxDD | 8.08% / 0.71 / -16.1% | 7.03% / 0.39 / -42.7% |
| validation CAGR / SR / maxDD | 8.86% / 0.63 / -13.2% | 13.47% / 0.66 / -26.2% |

---

## 7. Deployment

`USTEC_RiskManagedLong.mq5`, compiled clean against the MetaTrader 5 build on
this machine.

- Attach to **USTEC**, any timeframe (it reads M30 bars itself).
- It acts once per session, in a 30-minute window after 09:30 New York, keyed off
  `TimeGMT()` and an explicit US DST rule - **not** off server time, because the
  broker has moved this instrument's UTC hours twice while never moving its New
  York hours.
- Lots come from `SYMBOL_TRADE_CONTRACT_SIZE` at runtime rather than a constant,
  which also insulates it from the project's acknowledged uncertainty about
  USTEC's contract size (`symbols.py` calls it "inferred, not published").
- State is kept in `GlobalVariable`s keyed by magic number, so a terminal restart
  does not force a spurious rebalance.
- Safety: spread guard, swap guard, and an equity-drawdown kill switch that
  flattens and stops.

**Before going live**, in order:

1. Read the account's actual USTEC swap and set `InpSwapCheckBps`. If it is above
   about 3 bps/night, the case for this weakens sharply.
2. Confirm the contract size and minimum lot support the intended equity. At
   0.01 lot minimum and a ~23,000 index level, the smallest position is ~$230 of
   notional, so weight resolution on a small account is coarse.
3. Paper trade for at least a month and check that the expert's logged weight
   matches what `scripts/backtests/backtest_risk_managed_long.py` says it should be.
4. Decide `InpTargetVolPct` from risk appetite, not from this report.

## 8. What would falsify this

- **Volatility stops clustering.** The block correlation is already down from
  +0.45 to +0.15 across the three periods. If it reaches zero, the sizing rule
  becomes noise and only the trend gate is left.
- **A gap through the gate.** The gate reads a daily close. A crash that happens
  overnight is taken at full size; nothing here protects against that, and the
  test period contained no such event.
- **Swap repricing.** A move to 4-5 bps/night removes the entire return.
- **Live drawdown beyond -20%.** Worst observed is -15.1% (dev). A materially
  worse live drawdown means the volatility forecast has stopped working, and the
  kill switch at -25% is the backstop, not the plan.

The honest expectation to hold: this is a long Nasdaq position with a risk
overlay. Over a full cycle it should return somewhat less than buy and hold and
draw down roughly half as much. Anyone expecting it to make money when the index
falls has misread it.
