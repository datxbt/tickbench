"""Liquidity sweeps and order blocks, from UAlgo's *Price Action Toolkit Lite*.

The source is a TradingView indicator (Pine v6, CC BY-NC-SA 4.0). It draws;
it does not trade. Two of its four components make a claim a trade can test,
and this module turns each into one, with every choice the indicator leaves
open made explicit and swept rather than assumed.

Liquidity sweep
---------------
``ta.pivothigh(high, 30, 30)`` marks a level, confirmed 30 bars after the
pivot. At most seven live levels per side; pushing an eighth drops the oldest
untested. A level is *taken* by the first bar whose high trades above it, and
is retired either way. If that same bar **closes back below** the level the
indicator prints an ``x`` - a stop run that failed. Mirror for pivot lows.

Trade: the ``x`` is a reversal call, so enter against the sweep **at market
on the sweep bar's close** (the standing quote at the bar's end). The stop is
the sweep bar's own wick extreme - the one price that says the reversal was
wrong - or a quarter ATR beyond it.

Order block
-----------
A one-sided zigzag (``high[9] >= highest(high, 9)``, trend flips on the
opposite swing) records swing highs and lows. A **close** beyond the latest
swing low is a bearish break of structure (``CHoCH`` if the previous break was
up, ``BoS`` if it was also down); the bearish order block is then the highest
high in the bars *after* that swing low up to the break bar, drawn as the zone
``[max - ATR14, max]``. Mirror for bullish. Only the newest two blocks per side
are shown; a shown block is deleted when price closes through its far edge.

Trade: the standard reading - a **resting limit at the zone's near edge**, in
the block's direction, live while the block is shown and not yet deleted. The
stop is the far edge (the indicator's own deletion line, so risk is one ATR)
or half an ATR beyond it.

Deliberate departures from the Pine source
------------------------------------------
* **Bar-count lookback.** The script converts a time difference into a bar
  count with ``(time - swingTime - dt) / dt``, where ``dt`` is the *current*
  bar's gap to the previous one. Across a weekend or a session break that
  count is wrong in both directions. The intended window - bars after the
  swing through the break bar - is implemented by index.
* **No re-arming.** A block pushed out of the newest two is hidden. In Pine it
  can reappear if a newer one is deleted; here its order is cancelled the first
  time it is hidden and never re-armed. The block still occupies its place in
  the list, so which blocks are shown matches the script bar for bar.
* **Order life.** Pine keeps a shown block forever; the order here dies after
  ``ob_valid_bars`` bars. Its bind rate is reported.
* **Pivot ties.** Pine does not document its tie rule. A pivot high here is
  strictly above the ``L`` bars on its left and at least as high as the ``L``
  on its right; with float mid prices ties are rare.

Timeframes
----------
One-minute bars to one day. Up to an hour the grid is UTC, which is the same
grid as New York because the offset is whole hours. From 2h up bars are cut
on the **17:00 New York day** - the FX market's day, the broker's rollover and
the halt on the two instruments that have one - so D1 has no Sunday stub and
H4 bars start at 17:00, 21:00, 01:00, 05:00, 09:00 and 13:00 New York.

Prices and fills
----------------
Bars are OHLC on the mid, so every level is a mid quantity. Fills are not: a
long buys the ask and sells the bid. Market legs (sweep entries, stops, time
exits, a block that is already inside its zone when it forms) pay the measured
slippage; resting limits (block entries, targets) fill at their price or not
at all. A market entry takes the **standing quote** at the arm instant - the
last quote at or before it - because the feed prints only on change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from ..costs import CostModel
from ..loader import SPLITS, load_bars, load_ticks
from ..symbols import SymbolSpec, get_spec

US = 1_000_000
NY = "America/New_York"

INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "20m": 20,
    "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}
ALL_INTERVALS: tuple[str, ...] = tuple(INTERVAL_MINUTES)
FAMILIES: tuple[str, ...] = ("sweep", "ob")

# Exits. ``s0`` is each family's natural stop (the sweep bar's wick, the
# block's far edge); ``s1`` the same with an ATR buffer. Targets are R
# multiples of the stop in use; ``hold`` has none and ends at the stop or the
# clock. The indicator nominates no exit, so all eight are resolved.
STOP_KEYS: tuple[str, ...] = ("s0", "s1")
TARGET_R: dict[str, float] = {"t1": 1.0, "t2": 2.0, "t3": 3.0}
HOLD = "hold"
TARGET_KEYS: tuple[str, ...] = (*TARGET_R, HOLD)
EXIT_KEYS: tuple[str, ...] = tuple(f"{s}_{t}" for s in STOP_KEYS for t in TARGET_KEYS)

# Order blocks only: target the opposite-side block on the chart - its near
# edge (``obn``, the first price where it starts) or its far edge (``obf``,
# through the whole block). Resolved only where the chart shows one.
OB_TARGETS: dict[str, str] = {"obn": "tp_near", "obf": "tp_far"}
OB_EXIT_KEYS: tuple[str, ...] = tuple(f"{s}_{t}" for s in STOP_KEYS for t in OB_TARGETS)
ALL_EXIT_KEYS: tuple[str, ...] = EXIT_KEYS + OB_EXIT_KEYS

FORWARD_BARS: tuple[int, ...] = (1, 5, 20)
FILL_REASONS: tuple[str, ...] = ("filled", "hidden", "invalidated", "expired", "no_tape", "bad_risk")


def fair_rate(target_r: float) -> float:
    """Strike rate a driftless walk gives a target ``k`` R away against a 1R stop.

    First passage between two fixed levels: ``1 / (1 + k)``. At zero cost it is
    also the break-even strike rate, so it is the null every exit is read
    against.
    """
    return 1.0 / (1.0 + target_r)


@dataclass(frozen=True)
class ToolkitConfig:
    """The indicator's defaults, and every choice it leaves open."""

    zigzag_len: int = 9
    liquidity_len: int = 30
    atr_len: int = 14
    max_levels: int = 7
    ob_show: int = 2
    ob_store: int = 20
    ob_valid_bars: int = 100
    """Bars a block's resting order stays live if it is never hidden or deleted."""
    max_hold_bars: int = 60
    """A filled position's life cap, in bars of its own timeframe."""
    sweep_stop_buffer_atr: float = 0.25
    ob_stop_buffer_atr: float = 0.5
    control: str = "none"
    """``none`` - the setups. ``matched`` - the mechanics-only null: size-matched
    ordinary bars given the same geometry. For sweeps, a random bar entered at
    market with the stop at its own wick; for blocks, a zone built by the same
    rule (extreme of the bars since the last opposite swing, one ATR deep) at a
    bar with **no** break of structure, run through the same show/delete
    machinery. Direction is a hash coin in both."""
    fade: str = "none"
    """Addendum B. ``flip`` - reverse every order and mirror its stops through
    the entry; ``mirror`` - reverse it and swap the original stop and target.
    Either way a block's resting limit becomes a stop order at the same price."""
    skip_break_hours: bool = True
    """Drop sub-hour signals armed inside the maintenance break (bars built from
    a handful of ticks). Hourly and slower bars close at the break by design."""
    lots: float = 0.01

    def buffer(self, family: str) -> float:
        return self.sweep_stop_buffer_atr if family == "sweep" else self.ob_stop_buffer_atr


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------

BAR_COLS = ["ts", "ts_open", "open", "high", "low", "close", "n_ticks"]


def _from_broker(expr: pl.Expr) -> pl.Expr:
    """Broker-clock (New York + 7h, naive) back to UTC."""
    return ((expr - pl.duration(hours=7))
            .dt.replace_time_zone(NY, ambiguous="earliest", non_existent="null")
            .dt.convert_time_zone("UTC"))


def build_bars(base: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Bars at ``interval`` from right-edge-labelled 1m bars.

    ``ts`` is the bar's *nominal* end - the instant a trader knows it closed -
    not its last observed minute, which would act on the knowledge that no
    further quote arrived.
    """
    m = INTERVAL_MINUTES[interval]
    if m == 1:
        return base.select(BAR_COLS)
    agg = dict(open=pl.col("open").first(), high=pl.col("high").max(),
               low=pl.col("low").min(), close=pl.col("close").last(),
               n_ticks=pl.col("n_ticks").sum())
    lazy = base.lazy().sort("ts_open")
    if m <= 60:
        return (lazy.group_by_dynamic("ts_open", every=f"{m}m", closed="left", label="left")
                .agg(**agg)
                .with_columns(ts=pl.col("ts_open").dt.offset_by(f"{m}m"))
                .select(BAR_COLS).collect())
    every = "1d" if m == 1440 else f"{m}m"
    bk = (pl.col("ts_open").dt.convert_time_zone(NY).dt.replace_time_zone(None)
          + pl.duration(hours=7))
    return (lazy.with_columns(_bk=bk)
            .group_by_dynamic("_bk", every=every, closed="left", label="left")
            .agg(**agg)
            .with_columns(ts_open=_from_broker(pl.col("_bk")),
                          ts=_from_broker(pl.col("_bk").dt.offset_by(every)))
            .drop_nulls("ts")
            .select(BAR_COLS).collect())


def atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """``ta.atr``: Wilder's RMA of the true range, seeded with an SMA."""
    size = high.size
    out = np.full(size, np.nan)
    if size < n:
        return out
    prev = np.r_[np.nan, close[:-1]]
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = high[0] - low[0]
    val = float(tr[:n].mean())
    out[n - 1] = val
    a = 1.0 / n
    for i in range(n, size):
        val += a * (tr[i] - val)
        out[i] = val
    return out


def pivot_centres(x: np.ndarray, L: int, high: bool) -> np.ndarray:
    """Indices ``c`` of ``ta.pivothigh/low(x, L, L)`` pivots, confirmed at ``c + L``."""
    if x.size < 2 * L + 1:
        return np.empty(0, dtype=np.int64)
    w = sliding_window_view(x, 2 * L + 1)
    centre = w[:, L]
    if high:
        ok = (centre > w[:, :L].max(1)) & (centre >= w[:, L + 1:].max(1))
    else:
        ok = (centre < w[:, :L].min(1)) & (centre <= w[:, L + 1:].min(1))
    return np.nonzero(ok)[0].astype(np.int64) + L


def _hash_u(ts_us: np.ndarray, mult: int, mod: int) -> np.ndarray:
    """A deterministic uniform draw from a timestamp - stable across runs and subsets."""
    return ((ts_us // 1000) * mult % mod) / float(mod)


# --------------------------------------------------------------------------
# Liquidity sweeps - pure bar arithmetic
# --------------------------------------------------------------------------

def sweep_events(h: np.ndarray, l: np.ndarray, c: np.ndarray, L: int = 30,
                 max_levels: int = 7) -> list[tuple]:
    """Every ``x`` the indicator prints: ``(bar, direction, level, n_levels, age)``.

    ``direction`` is the trade's: -1 after a swept pivot high, +1 after a swept
    pivot low. ``level`` is the most extreme level swept on the bar.
    """
    ph = {int(k) + L: int(k) for k in pivot_centres(h, L, True)}
    plo = {int(k) + L: int(k) for k in pivot_centres(l, L, False)}
    bear: list[tuple[float, int]] = []   # pivot highs: (value, centre)
    bull: list[tuple[float, int]] = []
    out: list[tuple] = []
    for t in range(h.size):
        k = ph.get(t)
        if k is not None:
            bear.append((float(h[k]), k))
            if len(bear) > max_levels:
                bear.pop(0)
        k = plo.get(t)
        if k is not None:
            bull.append((float(l[k]), k))
            if len(bull) > max_levels:
                bull.pop(0)
        if bear:
            hi = h[t]
            taken = [lv for lv in bear if hi > lv[0]]
            if taken:
                bear = [lv for lv in bear if not hi > lv[0]]
                swept = [lv for lv in taken if c[t] < lv[0]]
                if swept:
                    top = max(swept)
                    out.append((t, -1, top[0], len(swept), t - top[1]))
        if bull:
            lo = l[t]
            taken = [lv for lv in bull if lo < lv[0]]
            if taken:
                bull = [lv for lv in bull if not lo < lv[0]]
                swept = [lv for lv in taken if c[t] > lv[0]]
                if swept:
                    bot = min(swept)
                    out.append((t, 1, bot[0], len(swept), t - bot[1]))
    return out


# --------------------------------------------------------------------------
# Structure and order blocks
# --------------------------------------------------------------------------

@dataclass
class Structure:
    """The zigzag's state at every bar, and the order blocks it produced."""

    blocks: list[dict]
    last_hi: np.ndarray    # index of the latest swing high, -1 if fewer than two
    last_lo: np.ndarray
    is_break: np.ndarray


def structure(h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray,
              Z: int = 9) -> Structure:
    """Replay the indicator's zigzag, BoS/CHoCH and order-block creation."""
    n = h.size
    to_up = np.zeros(n, bool)
    to_down = np.zeros(n, bool)
    if n > Z:
        wh = sliding_window_view(h, Z + 1)
        wl = sliding_window_view(l, Z + 1)
        to_up[Z:] = wh[:, 0] >= wh[:, 1:].max(1)
        to_down[Z:] = wl[:, 0] <= wl[:, 1:].min(1)

    trend = 1
    hv: list[float] = []
    hi_idx: list[int] = []
    lv: list[float] = []
    lo_idx: list[int] = []
    draw_up = draw_down = False
    last_state: str | None = None
    blocks: list[dict] = []
    last_hi = np.full(n, -1, np.int64)
    last_lo = np.full(n, -1, np.int64)
    is_break = np.zeros(n, bool)

    for t in range(n):
        prev = trend
        if trend == 1 and to_down[t]:
            trend = -1
        elif trend == -1 and to_up[t]:
            trend = 1
        if t > 0 and trend != prev:
            if trend == 1:
                hi_idx.append(t - Z)
                hv.append(float(h[t - Z]))
                if len(lv) > 1:
                    draw_up = False
            else:
                lo_idx.append(t - Z)
                lv.append(float(l[t - Z]))
                if len(hv) > 1:
                    draw_down = False
        if len(lv) > 1 and not draw_down and c[t] < lv[-1]:
            kind = "CHoCH" if last_state in (None, "up") else "BoS"
            draw_down = True
            last_state = "down"
            s = lo_idx[-1]
            if s < t and np.isfinite(atr[t]):
                seg = h[s + 1:t + 1]
                j = int(seg.argmax())
                v = float(seg[j])
                blocks.append(dict(create=t, direction=-1, value=v, atr=float(atr[t]),
                                   top=v, bottom=v - float(atr[t]), swing=s,
                                   origin=s + 1 + j, kind=kind))
            is_break[t] = True
        if len(hv) > 1 and not draw_up and c[t] > hv[-1]:
            kind = "CHoCH" if last_state in (None, "down") else "BoS"
            draw_up = True
            last_state = "up"
            s = hi_idx[-1]
            if s < t and np.isfinite(atr[t]):
                seg = l[s + 1:t + 1]
                j = int(seg.argmin())
                v = float(seg[j])
                blocks.append(dict(create=t, direction=1, value=v, atr=float(atr[t]),
                                   top=v + float(atr[t]), bottom=v, swing=s,
                                   origin=s + 1 + j, kind=kind))
            is_break[t] = True
        last_hi[t] = hi_idx[-1] if len(hv) > 1 else -1
        last_lo[t] = lo_idx[-1] if len(lv) > 1 else -1
    return Structure(blocks=blocks, last_hi=last_hi, last_lo=last_lo, is_break=is_break)


def block_lifetimes(blocks: list[dict], c: np.ndarray, *, show: int = 2,
                    store: int = 20, valid_bars: int = 100,
                    snapshot: bool = False) -> dict[str, np.ndarray] | None:
    """Set each block's ``cancel`` bar and ``cancel_reason`` in place.

    The indicator's show/delete loop, bar by bar: the newest ``show`` blocks per
    side are shown and deleted on a close through their far edge; older ones
    are hidden. The order dies at the first of hidden, deleted, or
    ``valid_bars`` after creation.

    ``snapshot=True`` also returns what the chart shows after each bar: for
    each side, the near and far edges of the newest ``show`` blocks still in
    the list, newest first, as ``(n, show)`` arrays (NaN where there is none).
    This is exactly the script's display rule - a hidden block that resurfaces
    when a newer one is deleted *is* shown again - even though its order is not
    re-armed.
    """
    n = c.size
    for b in blocks:
        b["cancel"], b["cancel_reason"], b["live"] = min(b["create"] + valid_bars, n - 1), "expired", True
    by_bar: dict[int, list[dict]] = {}
    for b in blocks:
        by_bar.setdefault(b["create"], []).append(b)
    snap = None
    if snapshot:
        snap = {k: np.full((n, show), np.nan)
                for k in ("bull_near", "bull_far", "bear_near", "bear_far")}
    if not blocks:
        return snap
    lists: dict[int, list[dict]] = {1: [], -1: []}
    start = min(by_bar)
    for t in range(start, n):
        for b in by_bar.get(t, ()):
            side = lists[b["direction"]]
            side.append(b)
            if len(side) > store:
                old = side.pop(0)
                _kill(old, t, "hidden")
        for d, side in lists.items():
            counter = 0
            for i in range(len(side) - 1, -1, -1):
                b = side[i]
                if counter < show:
                    if (d == 1 and c[t] < b["value"]) or (d == -1 and c[t] > b["value"]):
                        _kill(b, t, "invalidated")
                        side.pop(i)
                    counter += 1
                else:
                    if not b["live"]:
                        break
                    _kill(b, t, "hidden")
        for b in by_bar.get(t - valid_bars, ()):
            if b["live"]:
                b["live"] = False   # the default cancel is already the expiry
        if snap is not None:
            for d, name in ((1, "bull"), (-1, "bear")):
                for j, b in enumerate(reversed(lists[d][-show:])):
                    # a bullish block is entered from above: its near edge is the top
                    snap[f"{name}_near"][t, j] = b["top"] if d == 1 else b["bottom"]
                    snap[f"{name}_far"][t, j] = b["bottom"] if d == 1 else b["top"]
    return snap


def _kill(b: dict, t: int, reason: str) -> None:
    """First death wins. An expired block keeps its place in the list - Pine
    never expires anything - but its order is already gone."""
    if b["live"]:
        b["live"] = False
        if t < b["cancel"]:
            b["cancel"], b["cancel_reason"] = t, reason


# --------------------------------------------------------------------------
# The signal frame - one row per order the account would place
# --------------------------------------------------------------------------

SIGNAL_SCHEMA: dict[str, pl.DataType] = {
    "interval": pl.String, "family": pl.String, "arm_idx": pl.Int64,
    "arm_us": pl.Int64, "direction": pl.Int32, "kind": pl.String,
    "entry_level": pl.Float64, "stop0": pl.Float64, "stop1": pl.Float64,
    "atr": pl.Float64, "bar_high": pl.Float64, "bar_low": pl.Float64,
    "close": pl.Float64, "cancel_us": pl.Int64, "cancel_reason": pl.String,
    "level": pl.Float64, "n_levels": pl.Int64, "level_age": pl.Int64,
    "depth_atr": pl.Float64, "reclaim_atr": pl.Float64,
    "break_kind": pl.String, "lookback": pl.Int64, "zone_dist_atr": pl.Float64,
    "tp_near": pl.Float64, "tp_far": pl.Float64,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=SIGNAL_SCHEMA)


def signals(bars: pl.DataFrame, family: str, cfg: ToolkitConfig, *,
            interval: str = "5m") -> pl.DataFrame:
    """Every order one family places on ``bars``, as a frame."""
    if bars.height < 3 * cfg.liquidity_len:
        return _empty()
    h, l, c = (bars[k].to_numpy().astype(float) for k in ("high", "low", "close"))
    ts = bars["ts"].cast(pl.Int64).to_numpy()
    atr = atr_rma(h, l, c, cfg.atr_len)
    if family == "sweep":
        rows = _sweep_rows(h, l, c, ts, atr, cfg)
    elif family == "ob":
        rows = _block_rows(h, l, c, ts, atr, cfg)
    else:
        raise ValueError(f"unknown family {family!r}")
    if not rows:
        return _empty()
    frame = pl.DataFrame(rows, schema_overrides=SIGNAL_SCHEMA, infer_schema_length=None)
    frame = frame.with_columns(interval=pl.lit(interval), family=pl.lit(family))
    for col, dt in SIGNAL_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dt).alias(col))
    frame = frame.select(list(SIGNAL_SCHEMA)).cast(SIGNAL_SCHEMA)
    return _fade(frame, cfg.fade).sort("arm_us")


def _sweep_rows(h, l, c, ts, atr, cfg) -> list[dict]:
    L = cfg.liquidity_len
    events = sweep_events(h, l, c, L, cfg.max_levels)
    if cfg.control == "matched":
        hit = np.zeros(h.size, bool)
        hit[[e[0] for e in events]] = True
        cand = ~hit & np.isfinite(atr) & (h > l)
        cand[: 2 * L] = False
        n_c = int(cand.sum())
        if not events or n_c == 0:
            return []
        rate = min(1.0, len(events) / n_c)
        pick = np.nonzero(cand & (_hash_u(ts, 2654435761, 1000003) < rate))[0]
        coin = _hash_u(ts[pick], 2246822519, 1000033) < 0.5
        events = [(int(t), -1 if s else 1, float("nan"), 0, 0) for t, s in zip(pick, coin)]
    rows = []
    buf = cfg.sweep_stop_buffer_atr
    for t, d, level, n_lv, age in events:
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        wick = h[t] if d == -1 else l[t]
        rows.append(dict(
            arm_idx=t, arm_us=int(ts[t]), direction=d, kind="market",
            entry_level=None, stop0=float(wick), stop1=float(wick - d * buf * a),
            atr=float(a), bar_high=float(h[t]), bar_low=float(l[t]), close=float(c[t]),
            level=None if np.isnan(level) else level, n_levels=n_lv, level_age=age,
            depth_atr=None if np.isnan(level) else float(-d * (wick - level) / a),
            reclaim_atr=None if np.isnan(level) else float(d * (c[t] - level) / a),
        ))
    return rows


def _block_rows(h, l, c, ts, atr, cfg) -> list[dict]:
    st = structure(h, l, c, atr, cfg.zigzag_len)
    # What the chart shows, bar by bar. The opposite-block target always comes
    # from the indicator's real blocks - for the null's zones too - so the null
    # differs from the setup in the entry zone and nothing else.
    snap = block_lifetimes(st.blocks, c, show=cfg.ob_show, store=cfg.ob_store,
                           valid_bars=cfg.ob_valid_bars, snapshot=True)
    blocks = st.blocks
    if cfg.control == "matched":
        cand = ~st.is_break & np.isfinite(atr) & (st.last_hi >= 0) & (st.last_lo >= 0)
        n_c = int(cand.sum())
        if not blocks or n_c == 0:
            return []
        rate = min(1.0, len(blocks) / n_c)
        pick = np.nonzero(cand & (_hash_u(ts, 2654435761, 1000003) < rate))[0]
        coin = _hash_u(ts[pick], 2246822519, 1000033) < 0.5
        blocks = []
        for t, bear in zip(pick, coin):
            t = int(t)
            a = float(atr[t])
            if bear:
                s = int(st.last_lo[t])
                if s >= t:
                    continue
                seg = h[s + 1:t + 1]
                j = int(seg.argmax())
                v = float(seg[j])
                blocks.append(dict(create=t, direction=-1, value=v, atr=a, top=v,
                                   bottom=v - a, swing=s, origin=s + 1 + j, kind="none"))
            else:
                s = int(st.last_hi[t])
                if s >= t:
                    continue
                seg = l[s + 1:t + 1]
                j = int(seg.argmin())
                v = float(seg[j])
                blocks.append(dict(create=t, direction=1, value=v, atr=a, top=v + a,
                                   bottom=v, swing=s, origin=s + 1 + j, kind="none"))
    if cfg.control == "matched":
        block_lifetimes(blocks, c, show=cfg.ob_show, store=cfg.ob_store,
                        valid_bars=cfg.ob_valid_bars)
    rows = []
    buf = cfg.ob_stop_buffer_atr
    for b in blocks:
        t, d, a = b["create"], b["direction"], b["atr"]
        near = b["top"] if d == 1 else b["bottom"]
        far = b["value"]
        tp_near, tp_far = _opposite_target(snap, t, d, near)
        rows.append(dict(
            arm_idx=t, arm_us=int(ts[t]), direction=d, kind="limit",
            entry_level=float(near), stop0=float(far), stop1=float(far - d * buf * a),
            atr=a, bar_high=float(h[t]), bar_low=float(l[t]), close=float(c[t]),
            cancel_us=int(ts[b["cancel"]]), cancel_reason=b["cancel_reason"],
            break_kind=b["kind"], lookback=t - b["swing"],
            zone_dist_atr=float(d * (c[t] - near) / a),
            tp_near=tp_near, tp_far=tp_far,
        ))
    return rows


def _opposite_target(snap: dict[str, np.ndarray], t: int, d: int,
                     entry: float) -> tuple[float | None, float | None]:
    """The nearest opposite-side block shown at bar ``t``, beyond ``entry``.

    A long from a bullish block targets the lowest shown bearish block whose
    near edge (its bottom) is above the entry; a short, the highest shown
    bullish block whose near edge (its top) is below it. Returns that block's
    near and far edges, or ``(None, None)`` when the chart shows none.
    """
    side = "bear" if d == 1 else "bull"
    near, far = snap[f"{side}_near"][t], snap[f"{side}_far"][t]
    ok = (near > entry) if d == 1 else (near < entry)
    ok &= np.isfinite(near)
    if not ok.any():
        return None, None
    idx = np.nonzero(ok)[0]
    j = idx[np.argmin(near[idx])] if d == 1 else idx[np.argmax(near[idx])]
    return float(near[j]), float(far[j])


# --------------------------------------------------------------------------
# The tape
# --------------------------------------------------------------------------

@dataclass
class _Tape:
    ts: np.ndarray   # int64 microseconds, ascending
    bid: np.ndarray
    ask: np.ndarray

    def after(self, when_us: int) -> int:
        """First tick at or after ``when_us``."""
        return int(np.searchsorted(self.ts, when_us, side="left"))

    def standing(self, when_us: int) -> int:
        """The quote standing at ``when_us``: the last tick at or before it, or -1."""
        return int(np.searchsorted(self.ts, when_us, side="right")) - 1


class _TickCache:
    """Months of ticks, each loaded once and dropped once no batch needs it.

    A D1 block can rest for a hundred bars and then be held for sixty, so one
    month of signals can need most of a year of tape. Re-reading that per
    month would load the corpus several times over; a rolling cache loads it
    once. ``hard_end_us`` is the split's own end - no trade in a split is
    resolved on prices from the next one.
    """

    def __init__(self, symbol: str, hard_end_us: int) -> None:
        self.symbol = symbol
        self.hard_end_us = hard_end_us
        self.months: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def tape(self, start_us: int, end_us: int) -> _Tape:
        end_us = min(end_us, self.hard_end_us)
        lo = datetime.fromtimestamp(start_us / US, tz=timezone.utc)
        hi = datetime.fromtimestamp(max(end_us, start_us) / US, tz=timezone.utc)
        keys, y, m = [], lo.year, lo.month
        while (y, m) <= (hi.year, hi.month):
            keys.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        for k in [k for k in self.months if k < keys[0]]:
            del self.months[k]
        for k in keys:
            if k not in self.months:
                self.months[k] = self._load(*k)
        parts = [self.months[k] for k in keys]
        return _Tape(ts=np.concatenate([p[0] for p in parts]),
                     bid=np.concatenate([p[1] for p in parts]),
                     ask=np.concatenate([p[2] for p in parts]))

    def _load(self, y: int, m: int):
        start = datetime(y, m, 1, tzinfo=timezone.utc)
        end = datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc)
        end = min(end, datetime.fromtimestamp(self.hard_end_us / US, tz=timezone.utc))
        empty = (np.empty(0, np.int64), np.empty(0), np.empty(0))
        if end <= start:
            return empty
        try:
            t = load_ticks(self.symbol, start=start, end=end, allow_test=True)
        except FileNotFoundError:
            return empty
        return (t["ts"].cast(pl.Int64).to_numpy(), t["bid"].to_numpy(), t["ask"].to_numpy())


def roll_instants(start: date, end: date) -> np.ndarray:
    """Every weekday 17:00 New York in ``[start, end]``, as UTC microseconds.

    The broker's daily roll. A position held across it pays or earns swap,
    which this quote feed cannot see; counting the rolls each trade crosses lets
    the report charge it afterwards, from the measured overnight basis.
    """
    days = pl.date_range(start, end, "1d", eager=True)
    frame = pl.DataFrame({"d": days}).filter(pl.col("d").dt.weekday() <= 5)
    r = ((pl.col("d").cast(pl.Datetime("us")) + pl.duration(hours=17))
         .dt.replace_time_zone(NY).dt.convert_time_zone("UTC"))
    return frame.select(r.alias("r"))["r"].cast(pl.Int64).to_numpy()


# --------------------------------------------------------------------------
# One order, resolved against the tape
# --------------------------------------------------------------------------

from .engulfing_quadrant import _first_cross, _taken_indices  # noqa: E402


def _fade(frame: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Turn every order into its opposite (Addendum B).

    ``flip`` reverses the direction and mirrors both stops through the entry
    reference, so the new trade has the original's shape on the other side.
    ``mirror`` reverses the direction and keeps every original level; the
    resolver then swaps the original stop and target (see :func:`_brackets`).
    In both, a block's resting limit becomes a stop order at the same price:
    the opposite of buying a pullback into a level is selling the move through
    it.
    """
    if mode == "none" or frame.is_empty():
        return frame
    if mode not in ("flip", "mirror"):
        raise ValueError(f"unknown fade {mode!r}")
    ref = (pl.when(pl.col("kind") == "market").then(pl.col("close"))
           .otherwise(pl.col("entry_level")))
    if mode == "flip":
        frame = frame.with_columns(
            stop0=2 * ref - pl.col("stop0"), stop1=2 * ref - pl.col("stop1"),
            tp_near=pl.lit(None, pl.Float64), tp_far=pl.lit(None, pl.Float64),
        )
    return frame.with_columns(
        direction=-pl.col("direction"),
        kind=pl.when(pl.col("kind") == "limit").then(pl.lit("stop")).otherwise(pl.col("kind")),
    )


def _brackets(sig: dict, cfg: ToolkitConfig, d: int, entry: float) -> list[tuple] | None:
    """Every exit as ``(key, stop price, target price or None)``.

    Normal and ``flip``: each stop with 1R/2R/3R targets measured from the
    fill, no target (hold), and for blocks the opposite block beyond the fill.
    ``mirror``: the other side of those same brackets - the original's target
    is the stop and its stop is the target, both fixed from the original
    reference price. ``None`` when a normal stop is not beyond the fill.
    """
    stops = {"s0": float(sig["stop0"]), "s1": float(sig["stop1"])}
    out: list[tuple] = []
    if cfg.fade != "mirror":
        if any(d * (entry - v) <= 0 for v in stops.values()):
            return None
        for sk, stop in stops.items():
            risk = d * (entry - stop)
            for tk, k in TARGET_R.items():
                out.append((f"{sk}_{tk}", stop, entry + d * k * risk))
            out.append((f"{sk}_{HOLD}", stop, None))
            for tk, col in OB_TARGETS.items():
                tpx = sig.get(col)
                if tpx is not None and np.isfinite(tpx) and d * (tpx - entry) > 0:
                    out.append((f"{sk}_{tk}", stop, float(tpx)))
        return out
    d0 = -d
    ref = float(sig["close"] if sig["kind"] == "market" else sig["entry_level"])
    for sk, orig_stop in stops.items():
        r0 = d0 * (ref - orig_stop)
        if r0 <= 0:
            continue
        for tk, k in TARGET_R.items():
            out.append((f"{sk}_{tk}", ref + d0 * k * r0, orig_stop))
        for tk, col in OB_TARGETS.items():
            tpx = sig.get(col)
            if tpx is not None and np.isfinite(tpx) and d0 * (tpx - ref) > 0:
                out.append((f"{sk}_{tk}", float(tpx), orig_stop))
    return [b for b in out if d * (entry - b[1]) > 0] or None


def _resolve(tape: _Tape, sig: dict, bar_ts: np.ndarray, cfg: ToolkitConfig,
             slip: float, commission_px: float, usd_per_px: float,
             rolls: np.ndarray) -> dict:
    """Work the order, then price every exit on the same fill.

    Tie-breaks go against the trade: a tick that satisfies a stop and a target
    is a stop, because one quote cannot say which came first inside itself.
    """
    d = int(sig["direction"])
    row = dict(sig)
    row["filled"] = False
    work = tape.ask if d == 1 else tape.bid     # the side an entry trades on
    out = tape.bid if d == 1 else tape.ask      # the side an exit trades on
    n = tape.ts.size
    s = tape.standing(int(sig["arm_us"]))
    if s < 0 or s + 1 >= n:
        row["fill_reason"] = "no_tape"
        return row

    kind, level = sig["kind"], sig["entry_level"]
    if kind == "limit":
        # rests on the favourable side of the market: marketable only if price
        # is already through it when the order goes in
        market = work[s] <= level if d == 1 else work[s] >= level
    elif kind == "stop":
        # rests on the unfavourable side: triggered once price trades to it
        market = work[s] >= level if d == 1 else work[s] <= level
    else:
        market = True
    if market:
        # A sweep's entry, or a block order already marketable when placed:
        # a market order at the standing quote, paying slippage.
        f, entry, otype, slip_e = s, float(work[s]) + d * slip, "market", slip
    else:
        deadline = min(max(tape.after(int(sig["cancel_us"])), s + 1), n)
        below = (d == 1) if kind == "limit" else (d == -1)
        f = _first_cross(work, s + 1, deadline, level, below=below)
        if f < 0:
            row["fill_reason"] = sig["cancel_reason"]
            return row
        if kind == "limit":
            entry, otype, slip_e = float(level), "limit", 0.0
        else:
            # a triggered stop is a market order: it fills at the quote that
            # triggered it, gap included, and pays slippage
            entry, otype, slip_e = float(work[f]) + d * slip, "stop", slip

    fill_us = int(tape.ts[f])
    fb = max(int(np.searchsorted(bar_ts, fill_us, side="left")), int(sig["arm_idx"]))
    last = bar_ts.size - 1
    hold_end = tape.after(int(bar_ts[min(fb + cfg.max_hold_bars, last)]))
    hold_end = min(max(hold_end, f + 1), n)

    brackets = _brackets(sig, cfg, d, entry)
    if not brackets:
        row["fill_reason"] = "bad_risk"
        return row

    mid_f = float((tape.bid[f] + tape.ask[f]) / 2)
    spread_f = float(tape.ask[f] - tape.bid[f])
    row.update(fill_reason="filled", filled=True, order_type=otype, fill_us=fill_us,
               fill_bar=fb - int(sig["arm_idx"]), entry=entry, entry_mid=mid_f,
               spread_fill=spread_f, slip_px=slip, commission_px=commission_px,
               day=datetime.fromtimestamp(fill_us / US, tz=timezone.utc).date())
    rt_px = commission_px + spread_f + slip_e + slip   # the round turn, in price

    # Barrier-free, cost-free: the signal itself, before any exit is chosen.
    for hbar in FORWARD_BARS:
        v = float("nan")
        if fb + hbar <= last:
            i = tape.standing(int(bar_ts[fb + hbar]))
            if i > f:
                v = d * (float((tape.bid[i] + tape.ask[i]) / 2) - mid_f) / mid_f * 1e4
        row[f"fwd{hbar}_bps"] = v

    risk_by_key = {k: d * (entry - stop) for k, stop, _ in brackets}
    base = risk_by_key.get("s0_t2") or next(iter(risk_by_key.values()))
    seg = out[f:hold_end]
    best = float(seg.max() if d == 1 else seg.min())
    worst = float(seg.min() if d == 1 else seg.max())
    row["mfe_r"] = d * (best - entry) / base
    row["mae_r"] = d * (entry - worst) / base

    # One risk per stop for the normal brackets; the mirror's stop differs by
    # target, so its per-stop columns describe the mirror of the 2R trade.
    for sk in STOP_KEYS:
        if cfg.fade != "mirror":
            risk = d * (entry - float(sig["stop" + sk[1]]))
        else:
            risk = risk_by_key.get(f"{sk}_t2")
            if risk is None:
                continue
        row[f"risk_{sk}"] = risk
        row[f"risk_bps_{sk}"] = risk / entry * 1e4
        row[f"cost_r_{sk}"] = rt_px / risk

    end_px = float(out[hold_end - 1]) - d * slip
    base_rolls = np.searchsorted(rolls, fill_us, side="right")
    stop_hit: dict[float, int] = {}
    for key, stop, tpx in brackets:
        risk = risk_by_key[key]
        if stop not in stop_hit:
            stop_hit[stop] = _first_cross(out, f, hold_end, stop, below=(d == 1))
        si = stop_hit[stop]
        if tpx is None:
            ti, t_px = -1, None
        elif d * (tpx - entry) <= 0:
            # a mirrored target the stop fill has already gapped past: the
            # limit is marketable at once and fills at the quote
            ti, t_px = f, float(out[f])
        else:
            ti, t_px = _first_cross(out, f, hold_end, tpx, below=(d == -1)), tpx
        if ti >= 0 and (si < 0 or ti < si):
            px, reason, x = t_px, "tp", ti
        elif si >= 0:
            px, reason, x = float(out[si]) - d * slip, "stop", si
        else:
            px, reason, x = end_px, "time", hold_end - 1
        net = d * (px - entry) - commission_px
        x_us = int(tape.ts[x])
        row[f"r_{key}"] = net / risk
        row[f"usd_{key}"] = net * usd_per_px
        row[f"reason_{key}"] = reason
        row[f"exit_us_{key}"] = x_us
        row[f"rolls_{key}"] = int(np.searchsorted(rolls, x_us, side="right") - base_rolls)
        if tpx is not None and (cfg.fade == "mirror" or key.split("_", 1)[1] in OB_TARGETS):
            # the payoff this target offers, in R: its fair coin is 1/(1+tpr)
            row[f"tpr_{key}"] = max(d * (tpx - entry), 0.0) / risk
        if cfg.fade == "mirror":
            row[f"risk_{key}"] = risk
            row[f"cost_r_{key}"] = rt_px / risk
    return row


# --------------------------------------------------------------------------
# The account: one position at a time per (interval, family)
# --------------------------------------------------------------------------

def sequence_many(trades: pl.DataFrame, exit_keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    """The trades a one-position-at-a-time account takes, for each exit rule.

    Each (interval, family) is its own account: a signal whose fill lands
    while that account's previous trade is open is not taken.
    """
    if trades.is_empty():
        return {k: trades for k in exit_keys}
    keep: dict[str, list[pl.DataFrame]] = {k: [] for k in exit_keys}
    for _, g in trades.group_by(["interval", "family"], maintain_order=True):
        g = g.filter(pl.col("filled")).sort("fill_us")
        if g.is_empty():
            continue
        for k in exit_keys:
            # An exit that does not exist for a row (no opposite block on the
            # chart) is not a trade under that rule - and must not block one.
            col = f"exit_us_{k}"
            if col not in g.columns:
                continue
            gk = g.filter(pl.col(col).is_not_null())
            if gk.is_empty():
                continue
            keep[k].append(gk[_taken_indices(gk["fill_us"].to_numpy(), gk[col].to_numpy())])
    return {k: (pl.concat(v).sort("fill_us") if v else trades.clear()) for k, v in keep.items()}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(symbol: str, cfg: ToolkitConfig | None = None, *,
        families: Sequence[str] = FAMILIES,
        intervals: Sequence[str] = ALL_INTERVALS,
        split: str = "dev", allow_test: bool = False,
        cost: CostModel | None = None, verbose: bool = False) -> pl.DataFrame:
    """Both families on one symbol, every timeframe, one split.

    One row per order placed, filled or not. :func:`sequence_many` reduces it
    to what an account takes.
    """
    cfg = cfg or ToolkitConfig()
    spec = get_spec(symbol)
    cost = cost or CostModel.from_profiles(symbol, split=split)
    slip_by_hour = {h: cost.slippage_pips(hour=h) * spec.pip for h in range(24)}
    sp = SPLITS[split]
    hard_end = datetime(sp.end.year, sp.end.month, sp.end.day, tzinfo=timezone.utc) + timedelta(days=1)

    base = load_bars(symbol, "1m", split=split, allow_test=allow_test,
                     columns=["ts", "ts_open", "open", "high", "low", "close", "n_ticks"])
    if base.is_empty():
        return pl.DataFrame()

    frames, bar_ts = [], {}
    for iv in intervals:
        bars = build_bars(base, iv)
        bar_ts[iv] = bars["ts"].cast(pl.Int64).to_numpy()
        for fam in families:
            found = signals(bars, fam, cfg, interval=iv)
            if cfg.skip_break_hours and spec.daily_break_utc and INTERVAL_MINUTES[iv] < 60:
                lo, hi = spec.daily_break_utc
                hour = (pl.col("arm_us") // (3600 * US)) % 24
                found = found.filter(~hour.is_between(lo, hi - 1))
            if verbose:
                print(f"  {symbol} {iv:>3} {fam:5}: {found.height:,} orders", flush=True)
            frames.append(found)
    del base
    sigs = pl.concat(frames).sort("arm_us")
    if sigs.is_empty():
        return pl.DataFrame()

    rolls = roll_instants(sp.start - timedelta(days=3), sp.end + timedelta(days=3))
    cache = _TickCache(symbol, int(hard_end.timestamp() * US))
    lots_px = (spec.contract_size or 1.0) * cfg.lots
    life = {"sweep": 0, "ob": cfg.ob_valid_bars}
    span = cfg.max_hold_bars + max(FORWARD_BARS) + 1

    sigs = sigs.with_columns(_ym=pl.col("arm_us") // US)
    out: list[pl.DataFrame] = []
    months = sorted({(d.year, d.month) for d in
                     sigs.select(pl.from_epoch("_ym").dt.date())["_ym"].to_list()})
    for y, m in months:
        m0 = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * US)
        m1 = int(datetime(y + m // 12, m % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * US)
        batch = sigs.filter((pl.col("arm_us") >= m0) & (pl.col("arm_us") < m1))
        if batch.is_empty():
            continue
        need = m1
        for iv, fam, idx in batch.select("interval", "family", "arm_idx").iter_rows():
            arr = bar_ts[iv]
            need = max(need, int(arr[min(idx + life[fam] + span, arr.size - 1)]))
        tape = cache.tape(m0 - 3 * 86400 * US, need + 3600 * US)
        if tape.ts.size == 0:
            continue
        rows = []
        for sig in batch.drop("_ym").iter_rows(named=True):
            hour = (sig["arm_us"] // (3600 * US)) % 24
            rate = 1.0 if spec.quote_ccy == "USD" else sig["close"]
            rows.append(_resolve(tape, sig, bar_ts[sig["interval"]], cfg,
                                 slip_by_hour[hour],
                                 spec.commission_pips(rate) * spec.pip,
                                 lots_px / rate, rolls))
        out.append(pl.DataFrame(rows, infer_schema_length=None))
        del tape, rows
        if verbose:
            print(f"  {symbol} {y}-{m:02d}: {sum(f.height for f in out):,} rows", flush=True)
    if not out:
        return pl.DataFrame()
    return pl.concat(out, how="diagonal_relaxed").sort("arm_us")
