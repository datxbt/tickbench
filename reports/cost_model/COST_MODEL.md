# Cost model

Generated from the measured spread and latency profiles over the trailing 3 months. Slippage assumes **0.5 x** the measured mid drift over a **250 ms** fill latency; that fraction is an assumption about the strategy, not a measurement, and every figure below moves with it.

## What a round turn costs

| symbol | spread | commission | slippage | total | | in bps | USD/lot |
| --- | ---: | ---: | ---: | ---: | :-- | ---: | ---: |
| EURUSD | 0.029 | 0.500 | 0.063 | **0.592** | pips | 0.515 | 5.92 |
| USTEC | 0.420 | 0.626 | 0.620 | **1.666** | pips | 0.569 | 1.67 |
| USDJPY | 0.043 | 0.804 | 0.207 | **1.053** | pips | 0.655 | 6.55 |
| XAUUSD | 8.739 | 7.000 | 13.423 | **29.161** | pips | 0.689 | 29.16 |

Share of the total:

| symbol | spread | commission | slippage |
| --- | ---: | ---: | ---: |
| EURUSD | 5% | 84% | 11% |
| USTEC | 25% | 38% | 37% |
| USDJPY | 4% | 76% | 20% |
| XAUUSD | 30% | 24% | 46% |

## How much of this is the slippage assumption

The adverse fraction is the one number here that cannot be measured from the tape, so the honest presentation is a band rather than a point. `0.0` is a fill uncorrelated with the move, `1.0` a strategy that chases.

| symbol | 0.0 (none) | 0.5 (default) | 1.0 (chasing) | spread of the band |
| --- | ---: | ---: | ---: | ---: |
| EURUSD | 0.460 | 0.515 | 0.570 | 1.24x |
| USTEC | 0.358 | 0.569 | 0.781 | 2.18x |
| USDJPY | 0.526 | 0.655 | 0.784 | 1.49x |
| XAUUSD | 0.372 | 0.689 | 1.005 | 2.71x |

## When to trade

Cost is not spread evenly across the day. Hours with no quotes at all - the daily maintenance break - are marked closed rather than cheap.

| symbol | cheapest hours | dearest hours | worst / best | closed |
| --- | --- | --- | ---: | --- |
| EURUSD | 03h 0.53, 04h 0.53, 02h 0.53 | 18h 0.59, 12h 0.60, 21h 3.21 | 6.09x | none |
| USTEC | 10h 1.34, 09h 1.37, 02h 1.39 | 20h 1.76, 14h 1.92, 13h 2.08 | 1.55x | 21h |
| USDJPY | 05h 0.89, 02h 0.91, 16h 0.91 | 13h 1.19, 20h 1.52, 21h 4.42 | 4.96x | none |
| XAUUSD | 20h 24.65, 04h 24.82, 23h 25.55 | 12h 32.48, 14h 32.52, 13h 33.09 | 1.34x | 21h |

Spread at the Sunday reopen against the weekday mean:

| symbol | weekday | Sunday | ratio |
| --- | ---: | ---: | ---: |
| EURUSD | 0.0292 | 0.3669 | 12.6x |
| USTEC | 0.4200 | 0.5405 | 1.3x |
| USDJPY | 0.0425 | 0.8791 | 20.7x |
| XAUUSD | 8.7385 | 9.3818 | 1.1x |

## Cost is not constant across the splits

Spreads have compressed since 2020, so a backtest run on the dev split with today's costs would be flattered. The cost model takes the same `split=` argument the loader does, and should always be given it.

| symbol | dev | validation | test |
| --- | ---: | ---: | ---: |
| EURUSD | 0.579 | 0.589 | 0.538 |
| USTEC | 1.213 | 0.708 | 0.567 |
| USDJPY | 0.710 | 0.719 | 0.661 |
| XAUUSD | 1.090 | 0.657 | 0.683 |

(bps per round turn, priced at each split's own mean close.)

## What execution speed is worth

| symbol | 50 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 1000 vs 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 0.483 | 0.496 | 0.515 | 0.533 | 0.562 | 1.16x |
| USTEC | 0.457 | 0.494 | 0.569 | 0.656 | 0.784 | 1.72x |
| USDJPY | 0.594 | 0.618 | 0.655 | 0.694 | 0.754 | 1.27x |
| XAUUSD | 0.533 | 0.593 | 0.689 | 0.797 | 0.944 | 1.77x |

## The turnover budget

Annualised cost drag, in basis points, at a given number of round turns per trading day. This is the number a strategy's gross return has to clear before it has made anything.

| symbol | 1/day | 5/day | 20/day |
| --- | ---: | ---: | ---: |
| EURUSD | 130 | 649 | 2595 |
| USTEC | 144 | 718 | 2870 |
| USDJPY | 165 | 826 | 3303 |
| XAUUSD | 174 | 868 | 3470 |

## What this model does not cover

- **Market impact.** The feed carries no size, so there is no way to measure how a larger order moves the price. `impact_pips_per_lot` exists and defaults to zero; at retail size that is probably right, and at any size where it is not, this model is the wrong tool.
- **Rejections and requotes.** Modelled as though every order fills. On market execution the practical equivalent is a worse fill, which the slippage term absorbs only in expectation.
- **Swap and financing.** Positions held overnight pay or receive swap, which is published per symbol and is not in this model. Anything holding past 21:00 UTC needs it added.
- **Two inferred contract sizes.** Gold's 100 oz lot and USTEC's single index point are inferred, not published. Every pip-denominated figure here scales linearly with them.

