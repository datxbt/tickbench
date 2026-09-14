# Round numbers on EURUSD, USDJPY and USTEC - evaluation

Subject: Osler (2003, 2005), read off a real FX dealer's order book. Take-profit
orders cluster *at* round numbers and stop-loss orders just *beyond* them, so
price should revert at a round level that holds and run through one that breaks.
The project tested this on gold only ([level-interaction.md](level-interaction.md)),
where the reversion half came out backwards. Osler's evidence is FX, so the three
untested instruments are the ones the paper is actually about.

Method: `qlab.levels` and `qlab.eventstudy` unchanged; only the grids are new.
5-minute bars, the gold study's thresholds, round levels only, dev and
validation. Driver: `scripts/research/osler_levels.py`.

| instrument | grids, fine -> coarse |
| --- | --- |
| EURUSD | 10 pips, 50 pips, the big figure (100 pips) |
| USDJPY | 0.10, 0.50, 1.00 |
| USTEC | 50, 100, 500 points |

**Verdict: rejected. All nine pre-registered primary cells fail - the best is
t = +0.70 per day, with the opposite sign on validation. The test split was not
spent.**

---

## 1. The primary cells

Pre-registered: coarsest grid x {touch, sweep, breach} x three instruments, net
bps summed within each day and tested across days, at 30 minutes. Pass needed
t >= 2.77 (Bonferroni over nine) and the same sign on both splits. Direction is
the mechanism's: away from the level for a touch or a sweep, with it for a
breach.

| instrument | kind | events | mid bps | net bps | t / day | dev t | val t |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | touch | 359 | +0.18 | -0.39 | -0.68 | +0.00 | -1.83 |
| EURUSD | sweep | 328 | +0.97 | +0.41 | +0.70 | +1.05 | -0.51 |
| EURUSD | breach | 842 | -0.15 | -0.73 | -1.78 | -2.03 | -0.05 |
| USDJPY | touch | 437 | +0.16 | -0.56 | -0.96 | +0.07 | -1.60 |
| USDJPY | sweep | 485 | -0.90 | -1.63 | -2.50 | -2.74 | -0.41 |
| USDJPY | breach | 1,193 | +0.20 | -0.57 | -1.20 | -1.48 | -0.05 |
| USTEC | touch | 239 | -2.22 | -3.40 | -1.42 | -0.29 | -2.12 |
| USTEC | sweep | 256 | -0.50 | -1.65 | -0.76 | +0.28 | -1.97 |
| USTEC | breach | 609 | +0.42 | -0.71 | -0.32 | -0.59 | +0.05 |

Against the 12-draw placebo (each event moved 1-5 days at the same clock time),
every real mid lies inside the placebo range.

## 2. Three things worth recording

- **The fine grids are the cost, and nothing else.** On the 10-pip, 0.10-yen and
  50-point grids every cell is strongly negative per day (t down to -17) with a
  mid near zero. Events arrive several times a day and each pays a round turn.
  That is the pause-bar and engulfing lesson again: frequency without edge is a
  cost schedule.
- **The one ordering that goes Osler's way is EURUSD sweeps.** Reversion after a
  sweep grows as the grid coarsens - at 120 minutes +0.35, +0.39 and +1.56 bps at
  the mid for 10 pips, 50 pips and the big figure. That is the shape the
  mechanism predicts, on the instrument it was documented on. It is also inside
  its own placebo band (real +0.97 against a placebo maximum of +1.27 at 30
  minutes), +0.92 per-day t net at 120 minutes, and it flips sign on validation
  at 30 minutes. A shape, not an effect.
- **USDJPY big-figure sweeps continue rather than revert** - -0.90 bps at the mid
  at 30 minutes and -2.44 at 120, dev per-day t -2.74. Backwards, exactly as gold
  was. Two of the four instruments now have the reversion half pointing the
  wrong way; none has it pointing the right way.

## 3. Reproduce

```bash
python scripts/research/osler_levels.py
```

Output lands in `reports/osler_levels/`.
