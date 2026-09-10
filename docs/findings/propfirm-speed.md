# Least time to a funded account

**Recommendation: the USTEC overlay at 1.0x weight cap, traded flat at 4.0x, with
a −3% daily circuit breaker and flat across the weekend.** Median time to a
funded account, counting the attempts that fail along the way, is **2.2 months
on validation and 3.7 months on dev**, at about **3 entry fees (~€1,450–1,690)**
and a **31–37% pass rate per attempt**.

| | dev | validation |
| --- | ---: | ---: |
| median time to funded | **3.7 months** | **2.2 months** |
| 75th percentile | 6.6 months | 3.9 months |
| 90th percentile | 10.0 months | 5.9 months |
| attempts (mean) | 3.1 | 2.7 |
| expected fees @ €539 | €1,688 | €1,447 |
| P(pass) per attempt | 31% | 37% |
| worst day in the sample | −4.26% | −3.20% |
| days breaching the 5% floor | **0** | **0** |

- Rules engine: `src/qlab/propfirm.py`, tested in `tests/test_propfirm.py`
- Driver: `scripts/research/propfirm_speed.py`
- Expert: `USTEC_RiskManagedLong.mq5` — `InpPropMode`, `InpPropFlat`, `InpFlatWeekend`

This supersedes the timeline in [`propfirm-sizing.md`](propfirm-sizing.md), which
answered a different question and answered it correctly: *which size is safest*.
The answer there was 0.50x cushion, near-certain, and 2.1–3.6 years. Nothing
about that has been shown wrong. What changed is the question.

**The strategy is unchanged and nothing about it was fitted here.** It was
selected in an earlier study on its own splits. What this report chooses is a
size, a breaker level and a weekend rule, on dev and validation. **The test
split was not read.**

---

## 1. What was searched for first, and not found

The request was for a better *edge*. Three families were tested before
concluding there isn't one in this corpus, and they are reported because a
negative result that isn't written down gets re-run.

**Unconditional time-of-day drift.** Mean minute-return summed by New York hour,
across USTEC and XAUUSD. No hour is stable between splits: USTEC's hour 00 runs
t = +2.4 in dev and −1.3 in validation; hour 14 is +2.3 and −0.7. There is no
slice of the clock that reliably pays.

**Conditional intraday predictability.** A pre-registered grid — the session's
return so far, and the overnight gap, at eight decision times, against the
return remaining to the close. USTEC's overnight gap looks like the real thing
in dev, with rho between +0.13 and +0.21 at *every* decision time and Sharpe
0.4–0.6. In validation the sign reverses completely: rho −0.02 to −0.08, Sharpe
−0.4 to −0.8. Nothing in the grid survives both splits.

**A better volatility forecast.** The overlay sizes on close-to-close volatility
while a challenge measures the worst *moment* of the day, so forecasting the
adverse excursion directly should size better. Nine forecasts — close-to-close,
excursion-based, range-based, various windows and EWMAs — all land at R² between
0.03 and 0.10, and all leave a standardised excursion whose 99th percentile is
5.2 to 7.4 times its median. The tail is the same whichever forecast is used.

So the edge is what it was: **Sharpe ≈ 0.95, and no more available.** Speed has
to come from somewhere else.

---

## 2. Two things the previous analysis got wrong

Both were immaterial at 0.50x and decisive at 4x, which is why they surfaced
only when the question changed.

### The statistic conditioned on winning

`block_bootstrap` reports `median_days_when_passed` — the time taken by the runs
that worked. Nobody experiences that number. A failed challenge is not the end
of the project; it is an entry fee and a few weeks. What is actually lived is
the time until the *first* success across however many attempts it takes.

That inverts the ranking, because **a failure that arrives quickly is cheap in
time**. Large sizes resolve fast in both directions. `time_to_funded` in
`propfirm.py` walks attempt after attempt until one sticks, charging days for
the failures and a restart, and it is what every number in this report uses.

### A circuit breaker was assumed able to stop a gap

`daily_records_from_minutes` measured each broker day from its own first quote,
which silently dropped the move between one session's last quote and the next
one's first — the nightly halt and the whole weekend. It dropped it from the
return *and* from the worst excursion. Worse, `run_phase` then let the −3%
breaker truncate that move as though a stop could be placed inside a closed
market.

That is the most flattering error a challenge simulation can make: it converts
the one risk leverage cannot manage into one that is managed for free. Both are
fixed, and `test_a_breaker_cannot_stop_a_gap` pins the behaviour.

USTEC's dropped gap has a standard deviation of 0.44% (dev) and reaches −5.03%
and −5.15% in the two splits. At 4x, −1.25% is a dead account.

---

## 3. Where the ceiling on size actually is

A breaker is a promise to act. A gap is the market refusing to let you. So the
largest carryable size is set not by volatility — the breaker handles that — but
by the worst reopen in the sample.

| split | weekends | worst gap | 1-in-1000 | worst day | size cap |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | held | −3.70% | −1.01% | −5.39% | **1.35x** |
| dev | flat | −1.02% | −0.81% | −3.45% | **4.92x** |
| validation | held | −2.15% | −1.79% | −4.47% | **2.33x** |
| validation | flat | −0.48% | −0.42% | −4.47% | **10.44x** |

Holding through weekends caps the size at 1.35x on dev. **Closing before the
weekend raises that ceiling by a factor of 3.6, and that is what makes the whole
plan possible.**

The justification for the weekend rule is not that weekends pay badly. They
might: the unconditional weekend leg is −7.8 bps in dev and −7.7 in validation,
about −4%/yr in both, and it survives moving 15, 30 and 60 minutes off the
reopen quote so it is not a spread artifact. But t is −1.6 and −0.7. **That is
not significant and is not claimed.**

The justification is shape. The weekend leg carries two to three times the
weeknight standard deviation (0.68–0.78% against 0.20–0.35%) and contains the
worst reopen in each split, while its mean is negative in both and significant
in neither. A challenge is judged on the worst moment, not the average one.

Holiday closures are not detected by the expert. They are 4 of 205 multi-day
shutdowns in dev and 3 of 80 in validation, so the Friday rule covers 98% of the
exposure.

---

## 4. The frontier

Flat sizing, −3% breaker, retries counted, weekends held (the weekend rule
changes the tail, not the median — see §5).

| | size | median | p75 | p90 | attempts | E[fees] | P(pass) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 1.0x | 14.2 mo | 23.6 | 35.1 | 1.3 | €722 | 74% |
| | 2.0x | 6.2 mo | 10.7 | 16.7 | 2.2 | €1,163 | 46% |
| | 3.0x | 4.5 mo | 7.8 | 12.0 | 2.6 | €1,415 | 38% |
| | **4.0x** | **3.4 mo** | 6.1 | 9.2 | 3.0 | €1,593 | 35% |
| | 5.0x | 3.0 mo | 5.5 | 8.5 | 3.2 | €1,745 | 33% |
| validation | 1.0x | 11.0 mo | 17.8 | 28.0 | 1.7 | €939 | 58% |
| | 2.0x | 4.2 mo | 7.2 | 11.1 | 2.3 | €1,264 | 43% |
| | 3.0x | 2.8 mo | 4.8 | 7.1 | 2.5 | €1,333 | 40% |
| | **4.0x** | **2.3 mo** | 4.0 | 6.4 | 2.8 | €1,521 | 35% |
| | 5.0x | 1.9 mo | 3.4 | 5.4 | 2.7 | €1,461 | 33% |

Time falls steeply to about 3x and then flattens, while fees keep climbing. Past
4x you are paying real money for weeks.

**Why 4.0x and not 5.0x.** Not because 4 optimises anything. With the weekend
rule on, 4.0x is the largest size at which **no day in either split would have
breached the 5% floor** — worst day −4.26% on dev, −3.20% on validation, zero
hits. At 5.0x dev takes one hit at −5.28%. The cap is a property of the data,
not a tuned parameter.

---

## 5. The recommended configuration

Flat 4.0x, −3% breaker, flat over the weekend:

| split | median | p75 | p90 | attempts | E[fees] | P(pass) | worst day | floor hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 3.7 mo | 6.6 | 10.0 | 3.1 | €1,688 | 31% | −4.26% | 0 |
| validation | 2.2 mo | 3.9 | 5.9 | 2.7 | €1,447 | 37% | −3.20% | 0 |

Against holding weekends, this costs about 0.3 months of median time on dev and
saves 0.1 on validation — **a wash on speed, and it removes the failure mode
that no stop can address.** The bootstrap cannot price that properly, because
the event happens once or twice in four and a half years; that is an argument
for the rule, not against it.

### Expert settings

```
InpMaxWeight      = 1.0      // down from 2.0: a challenge watches floating equity
InpPropMode       = true
InpPropFlat       = true     // flat sizing, not cushion
InpPropScale      = 4.0
InpPropInitial    = 100000   // or 0 to latch the balance at attach
InpDailyStopPct   = 3.0      // the breaker
InpFlatWeekend    = true
InpWeekendFlatMin = 1000     // 16:40 New York on Friday
```

Everything else stays at its shipped default. Compiles clean, 0 errors and
0 warnings.

---

## 6. How much of this is the edge

Very little of it, at these sizes, and the report would be dishonest without
saying so. A positive-drift index levered four times reaches +10% quickly
whether or not the strategy adds anything. The control is the same days,
demeaned — the shape and the volatility clustering kept, the drift removed.

| split | size | P real | P zero-drift | lift | median real | median zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 1.0x | 75% | 25% | **+50pp** | 14.3 mo | 34.0 mo |
| dev | 2.0x | 47% | 21% | +26pp | 6.2 mo | 11.4 mo |
| dev | 4.0x | 34% | 19% | +14pp | 3.4 mo | 4.5 mo |
| validation | 1.0x | 59% | 20% | **+38pp** | 10.9 mo | 29.6 mo |
| validation | 2.0x | 44% | 24% | +20pp | 4.3 mo | 7.7 mo |
| validation | 4.0x | 33% | 23% | +10pp | 2.3 mo | 3.0 mo |

**At 1x the edge does nearly all the work. At 4x it does very little.** A
strategy with no edge at all would reach a funded account in a median of 3.0 to
4.5 months at this size; the real thing manages 2.3 to 3.7.

That is not an argument against the plan, but it is the correct way to read it:
**the fee is buying compressed time, not a better chance.** The entry fee is the
price of the compression, and the reason the plan is still worth running is
§7 — the account you end up with is worth far more than the fees, *and its value
rests entirely on the edge that 4x sizing wastes.*

---

## 7. The size that gets funded is not the size that stays funded

A funded account runs the same 5% daily and 10% total floors. At 4x the breaker
fires 45 times a year on dev and 75 on validation, each costing 3.2% — well over
100% a year in stop losses. **Carrying 4x onto a funded account destroys it.**

Drop to 1.0–1.5x on day one:

| split | weekends | size | worst day | floor hits | gross %/yr | your 80% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev | held | 1.0x | −3.90% | 0 | 10.40% | €8,323 |
| dev | held | **1.5x** | **−5.76%** | **1** | 13.99% | €11,192 |
| dev | flat | 1.5x | −3.20% | 0 | 11.13% | €8,905 |
| validation | held | 1.5x | −3.42% | 0 | 16.60% | €13,278 |
| validation | flat | 1.5x | −3.20% | 0 | 18.64% | €14,911 |

The weekend rule earns its keep here rather than during the challenge. **At 1.5x
holding weekends, dev contains one day that would have ended the account.** With
the rule on there are none, in either split, at any size up to 1.5x — and on
validation it also *earns more* (18.64% against 16.60%), because the leg it
skips was negative there.

So: ~€1,500 of fees and two to four months to acquire an asset worth roughly
€8,900–14,900 a year at the 80% split — *if the edge persists*, which §6 shows
is a real edge at 1x, whatever it is at 4x.

---

## 8. Robustness

**Slippage on the breaker.** At 4x it fires on roughly a quarter of all days, so
the fill assumption is most of the plan. Tripling it from 0.2% to 0.6% of equity:

| split | size | P(pass) 0.2% → 0.6% | median 0.2% → 0.6% | E[fees] |
| --- | ---: | ---: | ---: | ---: |
| dev | 4.0x | 32% → 27% | 3.3 → 3.6 mo | €1,591 → €1,923 |
| validation | 4.0x | 35% → 28% | 2.3 → 2.6 mo | €1,525 → €1,909 |

Half a month and about €350. **The plan degrades gracefully**, which matters
more than the point estimate.

**Parallel entries.** Three accounts at once is not three independent draws —
the same strategy on the same market is the same path. Three at 4.0x is
indistinguishable from one (dev 3.5 → 3.5 months). Three at 2x/4x/6x does help,
because different sizes hit the floors at different times: dev 3.5 → 2.5 months,
validation 2.3 → 1.7. It costs about double, €3,234. Real, but expensive per
month saved.

---

## 9. What would make this wrong

**The tail is priced from one or two events.** The worst gap appears once in
977 dev days. The bootstrap cannot resample an event larger than the largest in
the sample, and index weekend gaps have historically been worse than anything
here. The unconditional worst gap in the corpus is −5.03% and −5.15%; the
strategy escaped both only because the trend gate had it flat. **At 4x, a −2%
weekend gap while long ends the attempt instantly, and the sample says nothing
useful about how often that happens.**

**A quarter of days ending in a market-order flatten is an operational claim,
not just a modelling one.** Test the breaker on a demo account until it has
fired several times before running it with money.

**FTMO's terms change.** Every number here is arithmetic on the 2025 published
rules. Re-read them before starting.

**The dev/validation gap is not small.** 3.7 months against 2.2. Dev contains
2022, when the trend gate sat flat for 202 consecutive sessions — nine and a
half months of making no progress at all while the calendar ran. That is the
single largest source of variance in the timeline, it is not diversifiable
within one instrument, and it is why the p90 is 10 months rather than 5.

**The test split is unspent.** It should stay that way until a configuration is
committed to; there is one read left and it is worth more later.
