# Liquidity Vacuum Fade - XAUUSD

Tick-level backtest on 283M Exness raw-spread quotes, 2020-01-29 to 2026-09-01. Fills at bid/ask from the tick that triggered them, commission $7/lot round turn, $0.12 slippage on entry and on stop-outs.

## Verdict

The signal has a **small positive edge at mid** ($+29.89 per trade on dev) and an **all-in cost of $109.73 per trade**. It does not clear its own execution, and the gap is not close.

Dev -71.9%, OOS -46.3%. Every sensitivity variant loses. The participation filter, which the specification identifies as the thing that makes this work, does not separate reverting moves from non-reverting ones.

## Does the hypothesis hold?

Forward mid move after each of 3,892 setups on dev, signed so positive means the fade would have profited. The participation filter is dropped here so setups can be bucketed by it.

| rho bucket | setups | fade @5min | fade @15min | share positive | mean move |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-0.6 **(spec trades these)** | 193 | $+0.0229 | $-0.0058 | 53.9% | $1.91 |
| 0.6-0.9 **(spec trades these)** | 919 | $+0.0102 | $+0.0620 | 52.7% | $1.89 |
| 0.9-1.2 **(spec trades these)** | 1,283 | $-0.0059 | $+0.0251 | 52.9% | $1.98 |
| 1.2-1.5 | 732 | $+0.0977 | $+0.1545 | 54.4% | $2.22 |
| 1.5-2 | 415 | $-0.0123 | $-0.1858 | 52.3% | $2.36 |
| 2-99 | 350 | $+0.2038 | $+0.1654 | 56.9% | $2.83 |

## The strategy as specified

| period | trades | net | CAGR | Sharpe | max DD | win rate | PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 900 | -71.9% | -26.9% | -4.51 | -70.5% | 43.2% | 0.42 |
| validation | 166 | -21.7% | -14.8% | -2.93 | -23.6% | 46.4% | 0.46 |
| test | 390 | -31.4% | -44.4% | -5.00 | -32.4% | 50.3% | 0.59 |
| OOS (2024-2026) | 556 | -46.3% | -24.8% | -3.61 | -46.1% | 49.1% | 0.54 |

## Where the money goes

| period | edge at mid | spread+slippage | commission | net | cost / edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | $+26,901 | $81,603 | $17,157 | $-71,859 | 3.7x |
| validation | $-1,332 | $15,841 | $4,514 | $-21,687 | -x |
| test | $+7,759 | $30,395 | $8,796 | $-31,432 | 5.1x |
| OOS (2024-2026) | $+4,736 | $39,641 | $11,402 | $-46,307 | 10.8x |

## Sensitivity (dev split)

| variant | trades | net | edge at mid /trade | cost /trade | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| as specified | 900 | -71.9% | $+29.89 | $109.73 | -4.51 |
| pseudocode reset rule | 2,014 | -95.5% | $+9.72 | $57.13 | -6.68 |
| enter immediately (no 40-tick wait) | 1,962 | -99.1% | $+10.40 | $60.91 | -7.17 |
| zero slippage | 902 | -39.9% | $+43.62 | $87.80 | -1.83 |
| zero slippage + zero commission | 902 | -12.5% | $+43.62 | $57.45 | -0.55 |
| V >= 2.5 (looser) | 974 | -75.9% | $+25.40 | $103.38 | -4.77 |
| V >= 5.0 (tighter) | 120 | -22.5% | $-31.52 | $156.38 | -2.46 |
| no participation filter | 1,337 | -80.9% | $+30.64 | $91.17 | -4.84 |
| rho <= 0.9 (tighter) | 455 | -46.0% | $+38.30 | $139.33 | -3.06 |
| no cluster rule | 1,272 | -81.4% | $+29.20 | $93.19 | -4.96 |

## Is slippage really worst in a vacuum?

Measured directly: the mid move over 250 ms after each of 891 entries, signed against the position. Positive means the fill got worse.

- mean **$-0.0084**, median $-0.0030
- adverse on only **45.0%** of entries
- the backtest charges $0.12, roughly 14x the measured magnitude

So the specification's worry is not supported for this signal at this horizon, and the base case is already far more punitive than the tape. The strategy still loses with slippage set to zero.

