# Using the library

How to read data and charge cost from Python, rather than through a script.

Reading data downstream - always through `qlab.loader`, never by globbing
parquet directly, so that the cleaning policy and the split guard apply:

```python
from datetime import timedelta
from qlab.loader import load_bars, load_ticks
from qlab.bars import resample_bars
from qlab.session import with_session_flags
from qlab.symbols import get_spec

# Bars are small enough to load eagerly
bars = load_bars("XAUUSD", "1m", split="dev")

# Indicators warm on history before the split, and that history is flagged
bars = load_bars("XAUUSD", "1m", split="validation", warmup=timedelta(days=5))
evaluable = bars.filter(~bars["is_warmup"])

# Coarser bars come from the 1m bars, not from re-reading ticks
hourly = resample_bars(bars, "1h")

# Ticks are big: prefer lazy, and let the date filter prune whole months
ticks = load_ticks("EURUSD", start="2024-06-03", end="2024-06-07")
lazy = load_ticks("XAUUSD", lazy=True)

# Flag the expensive windows rather than dropping them
flagged = with_session_flags(bars, get_spec("XAUUSD"))
```

Costing a backtest - always with the same `split=` the bars were loaded with,
since cost has moved by nearly a factor of two across this corpus:

```python
from qlab.costs import CHASING, NO_SLIPPAGE, CostModel, SlippageModel

model = CostModel.from_profiles("XAUUSD", split="dev")
print(model.describe())

# Per-bar cost columns for a vectorized backtest. side_cost_pips is the
# composable one: multiply it by |position change|.
costed = model.with_costs(load_bars("XAUUSD", "1m", split="dev"))

# Scalar interface for an event-driven engine - same arithmetic
model.round_turn_pips(hour=13, price=2400.0)
model.round_turn_bps(price=2400.0)
model.annual_drag_bps(price=2400.0, round_turns_per_day=5)

# Stress, for Stage 6
model.stressed(spread_multiplier=3.0, adverse_fraction=1.0, extra_slippage_pips=1.0)
CostModel.from_profiles("XAUUSD", split="dev", slippage=SlippageModel(latency_ms=1000))
```
