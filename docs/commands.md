# Command reference

Every entry point in `scripts/`, grouped the way the directory is.


```bash
# Convert raw CSVs to parquet (resumable; skips files already up to date)
python scripts/pipeline/convert_ticks.py                  # all symbols
python scripts/pipeline/convert_ticks.py -s XAUUSD --force
python scripts/pipeline/convert_ticks.py -w 8             # worker processes

# Assess data quality and regenerate reports/data_quality/DATA_QUALITY.md
python scripts/pipeline/quality_report.py -w 8

# Build bars from the converted ticks (resumable; ~20 s for all four symbols)
python scripts/pipeline/build_bars.py -w 8               # 1m bars, all symbols
python scripts/pipeline/build_bars.py -i 1m 5m -s XAUUSD

# Coarser bars by exact re-aggregation of the 1m partition, same monthly layout.
# Boundaries are UTC, so every interval divides a month and the files have no
# seam; intervals that do not divide a day are refused. (~5 s, all four symbols)
python scripts/pipeline/resample_bars.py -i 4h 1d
python scripts/pipeline/resample_bars.py -i 4h -s XAUUSD --force

# Whole-history 1d/4h caches the FX-ML replication reads, under bars_whole/.
# Filtered (thin weekend bars dropped), so deliberately not in the bars/ layout.
python scripts/pipeline/prepare_fx_bars.py

# Measure the cost profiles, then regenerate reports/cost_model/COST_MODEL.md
python scripts/pipeline/build_cost_model.py -w 8
python scripts/pipeline/cost_report.py
python scripts/pipeline/cost_report.py --adverse 1.0 --latency 500   # stress

# Strategies. Both refuse the locked test split unless it is asked for.
python scripts/backtests/backtest_session_breakout.py
python scripts/backtests/backtest_session_breakout_top8.py             # TOP8_2026, 0.02 lots, $0.30 cap, D1-D5
python scripts/backtests/backtest_session_breakout_top8.py --phase analyse   # tables from the cached tapes and grid
python scripts/research/tester_reconcile.py REPORT.xlsx --phase parse halt report detail   # MT5 tester report vs the tick model
python scripts/research/tester_compare.py A_DIR B_DIR                  # two tester reports, one change at a time
python scripts/backtests/backtest_pause_bar.py                         # rejected, all four
python scripts/backtests/backtest_pause_bar.py --control               # the geometry-only null
python scripts/backtests/backtest_pause_bar.py --placebo               # the faded-signal check
python scripts/backtests/backtest_pause_bar.py --sweep                 # parameter sensitivity
python scripts/backtests/backtest_engulfing.py                         # 1m -> 4h, all four
python scripts/backtests/backtest_engulfing.py --control               # the geometry-only null
python scripts/backtests/backtest_engulfing.py --no-reverse            # what reversing is worth
python scripts/backtests/backtest_engulfing.py --placebo               # the faded-signal check
python scripts/backtests/backtest_engulfing_quadrant.py                # sweep + 0.25-0.5 entry
python scripts/backtests/backtest_engulfing_quadrant.py --controls     # matched and no-sweep nulls
python scripts/backtests/backtest_engulfing_quadrant.py --variants     # fade, mirror, prior-bull
python scripts/backtests/backtest_risk_managed_long.py                 # dev + validation
python scripts/backtests/backtest_risk_managed_long.py --swap 1.5      # with financing
python scripts/backtests/backtest_risk_managed_long.py --stress        # 3x spread, adverse
python scripts/backtests/backtest_open_ema.py            # NY open EMA rule, USTEC
python scripts/backtests/backtest_open_ema.py --cached   # re-report from parquet
python scripts/research/eurusd_research.py                            # the EURUSD rejections
python scripts/research/eurusd_research.py --only rollover
python scripts/backtests/backtest_carry_harvest.py                     # USDJPY, rejected
python scripts/backtests/backtest_carry_harvest.py --swap-credit 1.47
python scripts/research/xauusd_research.py                            # gold, rejected
python scripts/research/xauusd_research.py --only drift fix
python scripts/research/level_interaction.py                          # gold levels, rejected
python scripts/research/level_interaction.py --only kinds direction
python scripts/research/level_interaction.py --save                   # event tapes + JSON
python scripts/research/propfirm_eval.py                              # FTMO two-step sizing
python scripts/research/propfirm_eval.py --test --fee 539
python scripts/research/propfirm_speed.py                             # least time to funded
python scripts/research/propfirm_speed.py --section frontier --paths 3000
python scripts/backtests/backtest_overnight_reversal.py                # CO-OC and family, rejected
python scripts/backtests/backtest_overnight_reversal.py --only dispersion   # the mechanism regressions
python scripts/backtests/backtest_overnight_reversal.py --only placebo      # the shuffled-signal null
python scripts/backtests/backtest_opening_range.py                     # the 5-minute ORB, seven passes
python scripts/backtests/backtest_opening_range.py --placebo-draws 60  # the shuffled-sign null distribution
python scripts/backtests/backtest_overnight_drift.py                   # the 02:00 drift and buy-the-dip
python scripts/backtests/backtest_structural_break.py                  # gold swing breaks, all four axes
python scripts/backtests/backtest_news_breakout.py                     # SSRN 5246516, rejected
python scripts/backtests/backtest_news_breakout.py --audit-only        # is the calendar real?
python scripts/backtests/backtest_news_breakout.py --sweep             # the TI threshold surface
python scripts/backtests/backtest_volume_spar.py                       # SSRN 5757622, half accepted
python scripts/backtests/backtest_volume_spar.py --symbols USTEC --splits dev
python scripts/backtests/backtest_structural_break.py --only exit      # the half-exit ablation
python scripts/backtests/backtest_structural_break.py --only modes     # ticks vs interpolated vs closes
python scripts/backtests/backtest_structural_break.py --only entry --entry-exits b5 --paths 1000
python scripts/backtests/backtest_vwap_ema.py                          # gold VWAP/EMA, rejected
python scripts/backtests/backtest_vwap_ema.py --placebo-draws 60       # the random-entry null
python scripts/backtests/backtest_decision_tree.py                     # decision trees, rejected
python scripts/backtests/backtest_decision_tree.py --symbols XAUUSD    # one instrument only
python scripts/research/macro_event_screen.py                         # FOMC, CPI/NFP, month-end - wave 1
python scripts/research/macro_event_confirm.py                        # wave 2, disjoint release days
python scripts/research/fx_fix_screen.py                              # Tokyo gotobi fix, WM/R fix
python scripts/research/gotobi_robustness.py                          # the gotobi battery, dev + validation
python scripts/research/gotobi_ticks.py                               # tick path and latency fills
python scripts/research/gotobi_test.py                                # locked split - already spent, FAIL
python scripts/research/osler_levels.py                               # round numbers, EURUSD/USDJPY/USTEC
python scripts/research/close_rebalance.py                            # leveraged-fund flow into the US close
python scripts/research/fix_benchmarks.py                             # SGE gold auctions, ECB reference rate
python scripts/research/discovery_batch1.py --placebos 20              # discovery H01-H04, dev only
python scripts/research/discovery_batch2.py grid                       # H05 USDJPY continuation, dev grid + gate
python scripts/research/discovery_batch2.py validate                   # H05 + H06, validation - already spent
python scripts/research/discovery_batch3.py dev                        # H08-H10: spread shocks, tick activity, weekend
python scripts/research/discovery_batch4.py dev                        # H12: boosted 30m forecast, 4 instruments, holdout gate
python scripts/research/discovery_batch4.py dev --horizon 120          # H13: the same at 120m
python scripts/backtests/backtest_orb15.py                            # 15-minute ORB, dev + validation
python scripts/backtests/backtest_orb15.py --test                     # its single locked-split run
python scripts/backtests/backtest_sweep_orderblock.py                 # UAlgo sweeps + order blocks, 1m -> D1, with the null
python scripts/backtests/backtest_sweep_orderblock.py -s USTEC -i 1h 4h 1d --no-null
python scripts/backtests/backtest_sweep_orderblock.py --from-tapes    # re-summarise, no tape walk
python scripts/backtests/backtest_sweep_orderblock.py -f ob --primary s0_obn   # Addendum A: target the opposite block
python scripts/backtests/backtest_sweep_orderblock.py --fade mirror          # Addendum B: the other side of every trade
python scripts/backtests/backtest_sweep_orderblock.py --fade flip            # Addendum B: reversed, same stop shape
python scripts/backtests/backtest_precision_sniper.py --dry-run       # count signals, resolve nothing
python scripts/backtests/backtest_precision_sniper.py --filters       # Precision Sniper, 12 TFs x 6 presets, with the null
python scripts/backtests/backtest_precision_sniper.py -s USTEC -i 1h 4h 1d -p Default
python scripts/backtests/backtest_precision_sniper.py --from-tapes    # re-summarise, no tape walk
```
