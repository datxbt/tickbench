# Price levels on gold: the reversion half is backwards, the breach half is the drift

**Verdict: rejected. No level-based strategy is proposed.** Three kinds of level
interaction — touch, sweep, breach — against seven families of level, at six
horizons, on 5-minute and 1-minute bars, over the dev and validation splits. Of
207 cells with 100 or more events, **zero clear a multiple-comparison-corrected
threshold** and six clear it in the *negative* direction. Reproduce with
`python scripts/research/level_interaction.py`.

**Gold's test split is still unspent.** Nothing survived dev, so nothing was
promoted, and no call in this study passes `allow_test`.

This was the eleventh hypothesis family tried on gold and the first with a
peer-reviewed order-book mechanism behind it. It failed differently from the
other ten, and the way it failed is the useful part.

---

## 1. Why this one, and why it was worth the budget

Osler (2003, 2005) has direct evidence from the RBS order book that take-profit
orders cluster **at** round numbers and stop-loss orders cluster **just beyond**
them. That is two opposite predictions from one book, separated by which side of
the level price closed on: reversion at a level that holds, a cascade through
one that breaks.

`xauusd-rejected.md` §5 left a standing condition — a future gold hypothesis
gets a clean final estimate "provided it is a genuinely different hypothesis and
not another configuration of the ten above." Level clustering qualifies. None of
the ten touched it, it has a mechanism rather than a chart pattern, and it is
the one family the research literature actually supports.

### The three kinds are a partition

A bar meets a level in exactly one of three ways. This is the whole design, and
it is enforced in `qlab.levels` rather than left to a convention:

| kind | definition | direction |
| --- | --- | --- |
| `touch` | reached the level within 0.10 ATR, did not trade through, closed back | away from the level |
| `sweep` | traded through by ≤ 0.25 ATR, closed back on the origin side | away from the level |
| `breach` | closed beyond by > 0.25 ATR, having started the bar on the other side | with the breach |

A bar nowhere near a level, or one that pierced further than a sweep's worth
without closing beyond, is not an event. No bar produces two kinds against one
level — `test_the_three_kinds_are_a_partition` pins it — because if it could,
one approach to one level would be counted twice in a table that reads as though
the rows were independent.

Levels come from the **previous** bar's close, never the current one, so which
level a bar is judged against cannot depend on what the bar did. A level must
also have been untested for 60 bars, which is what separates the first approach
from the fortieth. Thresholds were fixed before anything was measured; none was
swept to find these results.

| | dev | validation |
| --- | ---: | ---: |
| 5-minute bars | 277,307 | 105,608 |
| events | 11,101 | 5,255 |
| round turn | 1.052 bps | 0.716 bps |

---

## 2. The core result: reversion is backwards

Dev, 5-minute bars, net of the measured cost stack. `bps/day` is what an account
taking every event in the row makes in a day, and `t/day` is its t-statistic
across trading days — the honest denominator, because the per-event t counts one
trending afternoon as a dozen observations.

| kind | horizon | mid | net | bps/day | t/day |
| --- | ---: | ---: | ---: | ---: | ---: |
| touch | 30 min | −0.53 | **−1.73** | −4.06 | −3.11 |
| touch | 120 min | −0.02 | −1.17 | −2.67 | −1.29 |
| sweep | 30 min | −0.62 | **−1.78** | −5.20 | −4.05 |
| sweep | 120 min | −1.47 | **−2.56** | −7.29 | −2.86 |
| breach | 30 min | +0.25 | −0.98 | −5.64 | −2.07 |
| breach | 120 min | **+1.56** | **+0.41** | +2.33 | +0.42 |

**The reversion half of the mechanism points the wrong way on gold, and it does
so at the mid.** Touch and sweep are negative before any cost is charged, at
almost every horizon, in both splits — on validation the sweep runs −0.63 to
−3.04 bps at the mid, with t below −2 at four of the six horizons. A negative
reversion return
is a positive continuation return, so the study says one thing consistently:
**gold goes through its levels, it does not bounce off them.**

That is not a small finding for anyone reading price-action material about gold.
The "liquidity sweep" — a marginal new extreme followed by a reversal — is the
single most-taught intraday gold pattern, and taking it cost 5.2 bps a day on
dev and 7.3 bps a day at the two-hour horizon. It loses at the mid, so it is not
a cost problem. It is backwards.

---

## 3. The one thing that looked real

Breach continuation at 120 minutes, cut by level family, is the best result in
the study — and it is ordered exactly the way the mechanism predicts:

| family | n | mid | net | bps/day | t/day |
| --- | ---: | ---: | ---: | ---: | ---: |
| **round_100** | 201 | **+6.27** | +5.00 | +5.00 | +1.37 |
| **round_50** | 363 | **+5.93** | +4.78 | +4.86 | +1.70 |
| **round_10** | 1,789 | **+3.36** | +2.20 | +4.16 | +1.30 |
| psh | 1,387 | +0.87 | −0.35 | −0.59 | −0.37 |
| pdh | 459 | +1.77 | +0.55 | +0.55 | +0.32 |
| psl | 1,366 | −1.19 | −2.21 | −3.66 | −2.06 |
| pdl | 423 | −1.15 | −2.25 | −2.25 | −1.32 |

The coarser the round-number grid, the larger the effect — $100 beats $50 beats
$10 — and the effect is confined to round numbers, with previous-day and
previous-session extremes contributing nothing. That ordering was written into
the script as a test before the numbers were seen, it is harder to pass by
chance than any single cell, and it passed.

It also beats its control. Each event moved one to five days at the same time of
day, weekday and direction, twelve independent draws:

| horizon | kind | real (mid) | placebo mean | placebo min | placebo max |
| --- | --- | ---: | ---: | ---: | ---: |
| 120 min | breach | **+1.56** | +0.12 | −1.01 | +0.68 |
| 120 min | sweep | −1.47 | +0.29 | −1.13 | +1.43 |
| 120 min | touch | −0.02 | +0.45 | −0.95 | +1.92 |

The real breach sits outside the range of all twelve placebo draws. The touch
sits comfortably inside it.

So: a mechanism-backed prediction, correctly ordered across level families,
outside its placebo band, and positive net of the full cost stack. That is more
than any gold candidate except the LBMA fix has managed.

---

## 4. Why it is not an edge anyway

Three separate reasons, any one of which is sufficient.

**It is the drift.** A breach is directional by construction — an upward breach
is a long — so anything that made money holding gold appears here as a level
effect unless the sides are separated. Net bps per event, breaches only:

| | | dev long | dev short | val long | val short |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60 min | | +0.08 | −1.12 | **+1.30** | +0.26 |
| 120 min | | +0.48 | +0.34 | **+2.03** | −0.59 |
| 240 min | | −0.04 | −0.24 | **+2.00** | −1.71 |

In dev — where `xauusd-rejected.md` §1 established gold has **+0.42%/yr** of
drift in the hours a flat-overnight strategy can hold — the two sides agree and
neither is significant. In validation, where gold ran +19.4%/yr in liquid hours,
the long side carries everything and the short side is negative. This is finding
#1 of the previous gold study arriving by a new route: **validation is just the
bull**, and a level definition is a slow way to discover that.

**The t-statistic is mostly overlap.** Breaches arrive about three to a day and
up to thirteen on a trending day, and the correlation between events-per-day and
that day's mean return is **+0.32**. So an event-weighted mean quietly
overweights exactly the days the signal was going to work on. The dev 120-minute
breach cell is +0.41 bps per event at an event-level t of +0.75; the same trades
summed within each day and tested across 1,063 days give t **+0.42**. Nothing in
the study clears
t = 2 on the daily statistic, in either split, in either direction, at any
horizon.

**It does not survive being one of 207 cells.** Cutting by kind × family ×
volatility regime × horizon and keeping cells with at least 100 events gives 207
of them. A Šidák correction at α = 0.05 across 207 trials demands |t| ≥ 3.66:

```
cells with a positive net mean:            50
cells clearing an uncorrected t = +2:       1
cells clearing the deflated threshold:      0
cells clearing a deflated threshold short:  6
```

One cell out of 207 clears an uncorrected t = +2, which is fewer than the five
that chance alone would produce. Six clear the corrected threshold on the losing
side — the study's only statistically solid results are that touching and
sweeping levels lose money.

---

## 5. Two things that were checked and did not rescue it

**A shorter timeframe makes it worse.** The same definitions on 1-minute bars
give 15,067 events and **every single net cell is negative**, at every kind and
every horizon, with the breach at 5 minutes at −1.25 bps and t/day −6.43. This
matches the `pause_bar` finding exactly: at 1 minute the round turn is a large
fraction of anything a bar can offer, and gold's wide bar range does not help
when the *predictable* fraction of it is small.

**Conditioning does not sharpen it.** Volatility regime, approach speed and
session were each cut separately. The best-looking cell is the breach in the
London/NY overlap: +1.27 bps at the mid, t +2.03 — and **+0.04 bps net**, t
+0.06. It is the LBMA-fix result again in a different costume: an effect that
exists at the mid, is the size of the spread, and is gone at the prices a trade
actually gets.

---

## 6. A correction to the cost assumption this study started from

The research brief that prompted this work estimated gold's all-in round turn at
**$0.11–$0.27/oz** and argued that gold's large bar-range-to-cost ratio makes
1m/5m strategies viable. Against this project's own measurements:

| | brief | measured (`COST_MODEL.md`) |
| --- | --- | --- |
| spread | ~$0.16/oz | **$0.087/oz** |
| commission | $0.070/oz | $0.070/oz ✓ |
| slippage | "variable" | **$0.134/oz — 46% of the total** |
| round turn | $0.11–$0.27/oz | **$0.292/oz** |

The commission is right, the spread is overstated by about 1.8×, and the largest
single component is missing. The true round turn is above the top of the quoted
range. That matters because the brief's central recommendation — screen
hypotheses against a $0.11–$0.27 breakeven — would have passed several cells in
§2 that in fact lose.

---

## 7. What was kept

The negative result is most of the output, but two modules are durable and
neither is gold-specific:

- **`qlab.levels`** — level construction and the three-way classification, with
  the no-lookahead and freshness guarantees tested rather than asserted. Any
  instrument, any interval.
- **`qlab.eventstudy`** — forward-return distributions from an event table,
  always reported at the mid, at real bid/ask fills plus commission, and net of
  measured slippage. Plus the placebo control, the daily-clustered statistic and
  the deflated threshold. This is the LBMA-fix method from the previous gold
  study made reusable, and it sits *before* the engine on purpose: it maps
  signal → forward-return distribution, not signal → backtest, so no exit rule
  is chosen while an effect is still being measured.

Both were built for this study and both will outlive it. The next hypothesis on
any instrument gets a cost-honest screen for the price of an event table.

Event tapes and the headline numbers are in
`reports/strategies/level_interaction_events/` and
`reports/strategies/level_interaction.json`; every table above is a reduction of
those, so a later question can be traced back to a row.

---

## 8. What would change the answer

- **Not a threshold sweep.** The mid-level breach effect is +1.56 bps against a
  1.05 bps round turn. There is room for a *larger* effect to clear, but not for
  a differently-tuned version of this one: the ordering across grids is already
  the strongest form of the result, and the best cell in 207 does not reach an
  uncorrected t = 2 on the daily statistic.
- **A drift-neutral formulation would be the honest retry.** Everything positive
  here is long gold in a bull regime. A version that requires the long and short
  sides to agree — or that trades the round-number breach as a spread against
  gold's own trend — would be testing the level rather than the drift. It is a
  different hypothesis and would need its own budget.
- **More data, for the regime question.** Unchanged from the previous gold
  study: this corpus holds one financing regime and one bull regime, and 6.5
  years is not enough to say which is normal.

**Gold's test split remains unspent.** Eleven hypothesis families have now been
rejected without it being touched.
