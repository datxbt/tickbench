# The cost model


Full report: `reports/cost_model/COST_MODEL.md`, or the
[Stage 2 report page](https://claude.ai/code/artifact/542ca331-d6f6-4621-b476-eebb6d36b2fe).

Three components. Commission is published, so it is arithmetic. Spread is
observed, so the default reads it off the bar that was actually quoted. Slippage
is the one that has to be assumed, and Stage 2's main job was to stop that
assumption being invisible.

## Slippage was the missing term

Stage 0 costed a round turn as spread + commission, which silently sets slippage
to zero. It is not zero. Sampling the tape directly - anchor tick, prevailing mid
250 ms later, absolute difference - and assuming half that move is adverse:

| symbol | spread | commission | slippage | round turn | in bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 0.029 | 0.500 | 0.063 | **0.592** pips | 0.515 |
| USTEC | 0.420 | 0.626 | 0.620 | **1.666** pips | 0.569 |
| USDJPY | 0.043 | 0.804 | 0.207 | **1.053** pips | 0.655 |
| XAUUSD | 8.739 | 7.000 | 13.423 | **29.161** pips | 0.689 |

Slippage is **46% of the total on gold** and 37% on USTEC, against 11% on
EURUSD. So the omission was not uniform, and it fell hardest on the instruments
Stage 0 called cheapest.

**This reverses Stage 0's ranking.** On spread + commission alone gold was the
cheapest instrument to trade at 0.372 bps; with slippage it is the dearest at
0.689. All four still sit within a 1.3x band, so the broader conclusion holds -
cost is not the reason to pick one - but any argument that named gold as cheapest
was an artifact of leaving slippage out.

## The assumption is a band, not a number

What share of the measured drift is adverse depends on the strategy, not on the
tape: a chasing breakout entry eats most of it, a passive one need not. That
share is `SlippageModel.adverse_fraction`, and it is stated rather than chosen
for you:

| symbol | 0.0 (uncorrelated) | 0.5 (default) | 1.0 (chasing) | band |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | 0.460 | 0.515 | 0.570 | 1.24x |
| USTEC | 0.358 | 0.569 | 0.781 | 2.18x |
| USDJPY | 0.526 | 0.655 | 0.784 | 1.49x |
| XAUUSD | 0.372 | 0.689 | 1.005 | 2.71x |

Setting it to zero is a claim about the strategy, not a saving.

Sqrt-of-time scaling from bar range does **not** substitute for the measurement:
it underestimates real drift by 1.6x on EURUSD and 2.7x on USTEC, because
microstructure noise at 250 ms never accumulates into a one-minute range. Hence a
measured profile rather than a formula, and hence `latency_ms` must be one of the
horizons actually measured (50, 100, 250, 500, 1000 ms) rather than interpolated.

## Three things that follow

**Avoid 21:00 UTC.** The rollover hour costs **6.1x** the cheapest hour on
EURUSD (3.21 vs 0.53 pips) and 5.0x on USDJPY. Gold and USTEC halt outright, so
their hour 21 is closed rather than expensive - the model marks it, so that
taking a minimum over hours cannot mistake it for cheap. Sunday reopen spread
runs 12.6x (EURUSD) and 20.7x (USDJPY) the weekday mean, and only 1.1-1.3x on
gold and USTEC.

**Execution speed is worth far more on some instruments than others.** Going
from 50 ms to 1000 ms of latency raises the round turn by 1.77x on gold and 1.72x
on USTEC, but only 1.16x on EURUSD. A VPS earns its keep on the fast
instruments and barely registers on the majors.

**Cost is window-dependent, so the model takes `split=`.** Per round turn, in
bps:

| symbol | dev | validation | test |
| --- | ---: | ---: | ---: |
| EURUSD | 0.579 | 0.589 | 0.538 |
| USTEC | 1.213 | 0.708 | 0.567 |
| USDJPY | 0.710 | 0.719 | 0.661 |
| XAUUSD | 1.090 | 0.657 | 0.683 |

Dev-period trading on USTEC cost **2.1x** what it costs now. A backtest run on
the dev split with today's costs would be flattered by more than most edges are
worth, so `CostModel.with_costs()` refuses bars from outside the window the model
was measured on.

## The turnover budget

Annualised drag in bps, which is what a gross return has to clear:

| symbol | 1/day | 5/day | 20/day |
| --- | ---: | ---: | ---: |
| EURUSD | 130 | 649 | 2595 |
| USTEC | 144 | 718 | 2870 |
| USDJPY | 165 | 826 | 3303 |
| XAUUSD | 174 | 868 | 3470 |

One round turn a day costs 1.3-1.7% a year before a strategy has done anything.
Twenty costs 26-35%.

## Not modelled

Market impact (the feed carries no size), rejections and requotes, and overnight
swap. And gold's 100 oz lot and USTEC's single index point remain **inferred, not
published** - every pip-denominated figure above scales linearly with them.

