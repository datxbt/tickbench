# Sizing for a two-step prop challenge

**Recommendation: the USTEC overlay, capped at 1.0x weight, traded at 0.50x with
cushion sizing.** That passes both steps with probability **99.2% to 100%**
across dev, validation and the held-out split, with no failures at all on two of
the three, in a median of **2.1 to 3.6 years**.

The honest headline is the pair, not the first number. Near-certain, and slow.

| | dev | validation | test |
| --- | ---: | ---: | ---: |
| P(pass both steps) | **100.0%** | **100.0%** | **99.2%** |
| median time to funded | 3.6 yr | 2.1 yr | 2.5 yr |
| 90th percentile time | 8.5 yr | 6.8 yr | 7.2 yr |
| failure modes | none | none | daily loss 1% |

- Rules engine: `src/qlab/propfirm.py`, tested in `tests/test_propfirm.py`
- Driver: `scripts/research/propfirm_eval.py`
- Expert: `USTEC_RiskManagedLong.mq5`, `InpPropMode = true`

**The strategy is unchanged and nothing about it was fitted here.** It was
selected in an earlier study, on its own splits. What this report chooses is a
*size*, on dev and validation. The test split is read once, as confirmation.

---

## 1. Why this is a different problem

Every other report in this project measures a strategy run forever: Sharpe,
CAGR, Calmar. A challenge asks something else entirely — **reach +10%, then +5%,
without ever touching either of two floors** — and that changes what "good"
means in three specific ways.

**Position size stops being a preference and becomes the whole decision.** For a
diffusion with drift `mu` and volatility `sigma`, the probability of touching
`+a` before `-b` is governed by `theta = 2*mu/sigma²`. Scaling a strategy by `k`
scales `mu` by `k` and `sigma` by `k`, so **`theta` scales by `1/k`**: halving
the size roughly doubles `theta`. Smaller is strictly better for passing, and
the only thing it costs is time. Since FTMO dropped its calendar deadline, time
is nearly free — which makes the answer unusually clear-cut, and unusually far
from what the size of most challenge accounts suggests people do.

**The daily floor is a floating-equity rule, so a close-to-close backtest cannot
see it.** A day that dips 6% intraday and closes flat has failed the challenge
and passed the backtest. Every number here therefore uses a per-day *worst
excursion* built from minute bars alongside the close-to-close return.

**The total floor is static, not trailing.** FTMO measures the 10% against the
*initial* balance and it never moves. An account at +6% is 16% from the floor,
not 10%. That asymmetry is large, it is entirely in the trader's favour, and
§3 is about spending it.

---

## 2. What the leverage cap has to come down to, and why

The shipped strategy caps its weight at 2.0x, because in a quiet market the
volatility target asks for leverage. On a challenge that is a liability, and the
excursion table says so directly — this is the worst *floating* moment, not the
worst close:

| strategy `max_weight` | dev | validation | **test** |
| --- | ---: | ---: | ---: |
| 2.00 (as shipped) | −5.07% | −5.19% | **−7.61%** |
| 1.50 | −4.77% | −5.19% | −6.33% |
| **1.00** | **−3.67%** | **−4.49%** | **−5.33%** |
| 0.75 | −3.08% | −3.54% | −3.71% |

At the shipped cap, a single day in each split already reaches the 5% daily floor
unaided, and the test split contains a **−7.61%** day. Capping the weight at 1.0
is therefore the first change, and it costs almost nothing: the sensitivity table
in the USTEC report shows `max_weight` barely binds above 1.0 anyway.

Note what the test split added that the other two did not: **−5.33% at a 1.0 cap
is still a failing day.** It is why the recommendation is 0.50x rather than
0.60x, and it is a reminder that the worst day in the sample is not the worst day
there is.

---

## 3. Cushion sizing

Because the total floor is static, the room available is not the balance — it is
the distance to a fixed level. Size on that instead:

```
multiple = scale * (equity - floor) / (initial - floor)
```

At the start the cushion is full and the multiple is `scale`. At +5% it is
`1.5 * scale`. At −5% it is `0.5 * scale`. **The position goes to zero as the
cushion does**, which in continuous time makes the floor unreachable. Gaps break
that guarantee, which is why the simulation still reports occasional failures —
but it removes nearly all of them.

It dominates flat sizing everywhere, and not marginally:

| | dev | validation | test |
| --- | ---: | ---: | ---: |
| flat 0.50x | 90.1% | 83.5% | 95.9% |
| **cushion 0.50x** | **100.0%** | **100.0%** | **99.2%** |
| flat 0.75x | 75.7% | 68.9% | 86.3% |
| cushion 0.75x | 97.1% | 88.1% | 69.6% |

Flat sizing fails by total loss — 10% to 31% of paths. Cushion sizing at 0.50x
does not fail by total loss **at all**, in any split. The residual failure mode
changes character entirely: what is left is the daily rule, which cushion sizing
does not address because that floor resets each night regardless of how much
room the account has.

---

## 4. The frontier

Cushion sizing, strategy capped at 1.0x. `P(pass|resolved)` is the number that
matters when there is no deadline: of the runs that finished one way or the
other, how many finished by passing. Runs still going at the 12-year horizon are
censored, not failed.

| scale | dev | validation | test | median yrs (worst split) | failure modes |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.40x | 100.0% | 100.0% | 100.0% | 4.4 | none anywhere |
| **0.50x** | **100.0%** | **100.0%** | **99.2%** | **3.6** | daily 1% (test) |
| 0.60x | 100.0% | 97.8% | 89.8% | 3.1 | daily 2–10% |
| 0.75x | 97.1% | 88.1% | 69.6% | 2.6 | daily 2–28% |
| 1.00x | 83.4% | 67.2% | 45.9% | 1.9 | daily 14–49% |

0.50x is the recommendation because it is the **largest** size that still holds
100% on both selection splits. It was selectable without the test split; the test
column confirms rather than chooses.

The frontier is real and the choice is a preference, not a fact. Going to 0.75x
buys about a year and costs roughly 30 points of pass probability on the worst
split. Going to 1.00x halves the time again and takes the worst split under a
coin flip.

---

## 5. The economics

FTMO's fee on a 100k account is about €539, refunded on the first payout.

| sizing | P(pass) | expected fees | median yrs | funded, gross | your 80% share |
| --- | ---: | ---: | ---: | ---: | ---: |
| cushion 0.50x, dev | 100.0% | €539 | 3.6 | 4.9% | $3,946 |
| cushion 0.50x, validation | 100.0% | €539 | 2.1 | 6.4% | $5,134 |
| cushion 0.50x, test | 99.2% | €543 | 2.5 | 8.4% | $6,710 |
| cushion 0.75x, worst split | 69.6% | €774 | 2.6 | 12.6% | $10,066 |
| cushion 1.00x, worst split | 45.9% | €1,174 | 1.9 | 16.8% | $13,421 |

**The fee is not the cost. Time is.** At €539 refundable, even a 46% pass rate
costs about €1,174 in expected entry fees — trivial against a five-figure annual
share. What a failed attempt actually costs is the year or two spent on it.

That cuts both ways, and it argues *against* the most conservative setting: if
fees were the binding cost, 0.40x would be right. Because time is the binding
cost, sitting between 0.50x and 0.75x is defensible, and anyone who would rather
be funded in 18 months at a 70% chance than in 3 years at a 99% chance is making
a coherent choice rather than a mistake.

---

## 6. The stress test that matters

Every number above assumes the strategy's measured drift repeats. **It is not
statistically significant** — the mean session return has a t-statistic of 1.87,
1.10 and 1.22 on the three splits: positive everywhere, significant nowhere.
That is the assumption doing all the work, so it deserves to be stressed
directly.

Re-running with the drift removed and everything else kept — the volatility
clustering, the fat tails, the intraday lows:

| cushion 0.50x | P(pass, 12 yr) | P(pass\|resolved) | still running at 12 yr |
| --- | ---: | ---: | ---: |
| dev, as measured | 87% | 100.0% | 13% |
| **dev, zero drift** | **19%** | 100.0% | **81%** |
| validation, zero drift | 12% | 100.0% | 88% |
| test, zero drift | 22% | 97.4% | 77% |

Read the two right-hand columns together. With no edge, cushion sizing means you
**almost never fail — you just never finish.** The position shrinks toward the
floor and stalls above it, so the account grinds sideways for a decade instead of
blowing up.

That is the single most useful thing this study produced, and it is worth stating
plainly: **cushion sizing converts risk of ruin into risk of never completing.**
For a challenge with no deadline that is a good trade — the downside becomes the
fee plus the opportunity cost, rather than the fee plus a blown account. But it
means a high pass probability here is **not** evidence the strategy works. It is
what the sizing rule does to any return series with roughly this risk. The edge
still has to be real for the timeline to mean anything.

---

## 6b. The fast lane, and why three months is not on it

Asked for a median under three months, the answer is: **not with this strategy,
and the reason is arithmetic rather than effort.**

### What a three-month median requires

Passing +10% then +5% is 15.5% compounded. To have that as the *median* outcome
inside 63 trading days the strategy has to run near **62% a year**. The daily
floor then caps volatility: a 5% daily loss has to be rare, so 5% must sit at
several standard deviations of a daily move.

| 5% sits at | implied daily sd | annual sigma | **Sharpe needed** | P(a 5% day) over 63 days |
| ---: | ---: | ---: | ---: | ---: |
| 2.0 sigma | 2.50% | 39.7% | 1.56 | **76.5%** |
| 2.5 sigma | 2.00% | 31.7% | **1.95** | 32.5% |
| 3.0 sigma | 1.67% | 26.5% | **2.34** | 8.2% |
| 3.5 sigma | 1.43% | 22.7% | 2.73 | 1.5% |

**This overlay runs at a Sharpe of 0.95.** Even accepting a 5% day at 2.5 sigma -
which by itself breaches the daily floor on a third of 63-day runs - still needs
a Sharpe near 2.0. That is not a sizing problem, and no scaling fixes it:
scaling multiplies `mu` and `sigma` together and leaves the ratio exactly where
it started. Size trades time against probability *along* a frontier the Sharpe
fixes; it cannot move the frontier.

### The daily circuit breaker

At the sizes speed requires, **every** failure is the daily floor - 51% to 77%
of paths at 2.0x. Cushion sizing cannot help, because that floor resets nightly
however much room the account has. What helps is a rule aimed at it directly:
flatten at −3% on the day and stand down until tomorrow, turning an
uncontrolled breach into a known loss.

| at cushion 2.0x | pass rate | median months | failures |
| --- | ---: | ---: | --- |
| no breaker | 15.6–34.0% | 1.4–4.2 | daily loss 51–77% |
| **−3% daily breaker** | **80.1–100%** | 3.3–6.7 | total loss 0–11% |

It works, and it changes the failure mode from daily to total loss - repeated
stop-outs at −3.2% walking the account down to the 10% floor. `InpDailyStopPct`
switches it on and it is worth using at any size above 1.0x.

### The honest timeline

A two-month median *per successful attempt* is not a two-month median to being
funded, because failed attempts consume calendar too. Resampling the whole
journey - attempt, fail, buy another, repeat:

| breaker | scale | dev | validation | test | mean attempts | expected fees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| −3% | 2.0x | 42.0 mo | 23.3 mo | 40.7 mo | 2.3 | €1,217 |
| −3% | 3.0x | 44.0 mo | 11.0 mo | 15.5 mo | 2.9 | €1,542 |
| **−3% | 3.5x** | **22.9 mo** | **7.0 mo** | **15.3 mo** | 3.0 | €1,633 |

**The fastest defensible configuration has a median of 7 to 23 months to funded**,
depending which regime you get. Not three. Pushing size further does not help:
past 3.5x the pass rate falls faster than the speed rises, and expected calendar
goes back up.

Note also that buying several challenges in parallel does **not** fix this.
Running the same strategy on N accounts produces N perfectly correlated
outcomes - they all pass together or all fail together. Parallelism only buys
something if the accounts run genuinely different strategies, or at least
materially different sizes.

### What would actually get to three months

Sharpe roughly 2.0, against the 0.95 available. Two routes:

- **Better alpha.** Four instruments were searched for it in this project and
  none was found; the one deployed strategy is a risk overlay on the equity
  premium, not alpha.
- **Diversification, which is the realistic one.** Sharpe scales with the square
  root of the number of uncorrelated return streams, so 0.95 to 2.0 needs about
  **four to five** independent strategies of similar quality. There is one. That
  is the same conclusion the XAUUSD study reached from the other direction: the
  binding constraint on this project is breadth, not effort, and four
  instruments is too few to build it from.

Until that exists, three months is a wish rather than a plan, and any
configuration that appears to deliver it is doing so by taking a 10-to-30%
chance and calling it a median.

## 7. Deployment

`USTEC_RiskManagedLong.mq5`, compiled clean. Set:

```
InpPropMode       = true
InpPropScale      = 0.50      // 0.75 if you want speed over certainty
InpMaxWeight      = 1.0       // down from the shipped 2.0 - this one matters
InpPropInitial    = 0         // 0 latches the balance at attach
InpPropMaxLossPct = 10.0
InpTargetVolPct   = 15.0      // unchanged
InpDailyStopPct   = 0.0       // 3.0 for any scale above 1.0x - see 6b
```

For the fast lane instead: `InpPropScale = 3.5`, `InpDailyStopPct = 3.0`,
`InpPropCushionCap = 4.0`. Read §6b before choosing it.

Two implementation details worth knowing, because both were bugs first:

- **The starting balance is latched once, at attach.** Re-reading it after a loss
  would walk the floor down with the account, which is precisely what the static
  rule punishes.
- **The no-trade band compares *effective* weights**, strategy target times
  cushion multiple. The cushion moves with equity even on days the strategy's own
  target does not, and comparing raw targets would strand the position at
  yesterday's cushion — which is not what was simulated.

**Before starting a paid challenge:** run it on a demo or FTMO free trial for a
month and confirm the expert's logged effective weight matches
`scripts/research/propfirm_eval.py`. A sizing rule is easy to get subtly wrong, and the
cheapest place to find out is not a live challenge.

---

## 8. What can still go wrong

- **The edge may not be real.** §6 is the honest version: a 12% to 22% chance of
  finishing in twelve years if the drift is zero. Nothing in the pass rate
  distinguishes a working strategy from a well-sized one.
- **The worst day in the sample is not the worst day there is.** Test contained a
  −7.61% floating excursion at the shipped cap, worse than anything in dev or
  validation. The daily floor is the one rule cushion sizing cannot protect.
- **Swap.** The USTEC overlay holds overnight and its break-even financing rate
  is about 5 bps a night. At 0.50x the position is halved and so is the drag, but
  it comes straight off a timeline that is already measured in years.
- **The rules change.** These are FTMO's 2025 terms: 10% and 5% targets, 5%
  daily, 10% static total, four-day minimum, no deadline. A firm that
  reintroduces a 30-day limit inverts the entire conclusion — `PropRules` takes
  `max_days` for exactly that reason, and `scripts/research/propfirm_eval.py` will re-run
  the frontier under it in one line.
- **Prop accounts are not the same as capital.** A funded account is a contract
  with rules that can be enforced, changed or disputed. Nothing in this report
  addresses that, and it is not a small consideration.
