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

# Measure the cost profiles, then regenerate reports/cost_model/COST_MODEL.md
python scripts/pipeline/build_cost_model.py -w 8
python scripts/pipeline/cost_report.py
python scripts/pipeline/cost_report.py --adverse 1.0 --latency 500   # stress

# Strategies. Both refuse the locked test split unless it is asked for.
python scripts/backtests/backtest_session_breakout.py
python scripts/backtests/backtest_pause_bar.py                         # rejected, all four
python scripts/backtests/backtest_pause_bar.py --control               # the geometry-only null
python scripts/backtests/backtest_pause_bar.py --placebo               # the faded-signal check
python scripts/backtests/backtest_pause_bar.py --sweep                 # parameter sensitivity
python scripts/backtests/backtest_engulfing.py                         # 1m -> 4h, all four
python scripts/backtests/backtest_engulfing.py --control               # the geometry-only null
python scripts/backtests/backtest_engulfing.py --no-reverse            # what reversing is worth
python scripts/backtests/backtest_engulfing.py --placebo               # the faded-signal check
python scripts/backtests/backtest_risk_managed_long.py                 # dev + validation
python scripts/backtests/backtest_risk_managed_long.py --swap 1.5      # with financing
python scripts/backtests/backtest_risk_managed_long.py --stress        # 3x spread, adverse
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
```
