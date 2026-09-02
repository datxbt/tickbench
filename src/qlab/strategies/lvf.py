"""Liquidity Vacuum Fade.

The hypothesis: a price move's information content is better measured per
*quote* than per *minute*. A three-dollar move in gold that took 400 ticks is
repricing; the same move on 80 ticks is a book with nothing in it. Fade the
second kind.

Three measurements over a rolling 100-tick window, each normalised against an
EMA(5000) baseline so that the thresholds mean the same thing in 2020 and 2026:

    V    = Move / EMA(Move)      how abnormal the move is        want >= 3.0
    E    = Move / Path           straight line, not chop         want >= 0.65
    rho  = Rate / EMA(Rate)      participation                   want <= 1.2

The third is the one doing the work. A large move on a *high* tick rate is news
and is left alone; a large move on a normal or low tick rate is the vacuum.

Implementation notes that matter for the result
-----------------------------------------------
**Baselines are lagged one tick.** ``EMA`` is shifted before the ratio is taken,
so the tick being judged is never part of the baseline it is judged against.
The effect at span 5000 is tiny, but "abnormal versus what came before" is the
claim, so that is what is computed.

**Windows spanning a session gap are refused.** Without this the strategy fires
at every Monday reopen and every end of the daily maintenance break: 100 ticks
straddling a weekend show an enormous Move and a near-zero tick Rate, which is
the signature the setup is looking for and the one place it means nothing. This
single guard is the difference between a plausible backtest and a fantasy.

**Cleaning policy is load-bearing here.** This strategy counts quotes, so what
counts as a quote decides the tick rate. The project default is right for it:
ticks byte-identical to their predecessor are dropped (a repeated quote is not
new information), while genuinely different quotes sharing a millisecond are
kept (they are separate updates that the feed's millisecond stamp merged).
Collapsing on timestamp instead would deflate the tick rate exactly during fast
moves - which is the numerator of the signal.

**Setups arrive in clusters.** V stays above its threshold for many consecutive
ticks, so only the first tick of a cluster arms; the rest are skipped until the
armed attempt resolves. This matches the reference pseudocode's ``if !armed``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import polars as pl

from .. import paths
from ..engine import Account, Fill, Trade, lots_for_risk
from ..loader import DEFAULT_POLICY, clean_ticks
from ..symbols import get_spec

US = 1_000_000  # microseconds per second


@dataclass(frozen=True)
class LVFParams:
    """Every threshold in one object, so a sensitivity run is one `replace()`."""

    window: int = 100
    ema_span: int = 5000

    v_min: float = 3.0
    e_min: float = 0.65
    rho_max: float = 1.2
    move_min: float = 1.20  # USD
    spread_max: float = 0.35  # USD

    arm_ticks: int = 40
    arm_max_ticks: int = 200
    extend_usd: float = 0.10
    extend_rule: str = "cancel"  # "cancel" (prose) | "reset" (pseudocode)

    stop_frac: float = 0.25
    stop_floor: float = 0.60  # USD
    tp1_frac: float = 0.50
    tp2_frac: float = 0.90
    time_stop_s: int = 45 * 60

    cluster_max: int = 2  # skip if this many same-direction vacuums already
    cluster_window_s: int = 30 * 60

    # 23:55-00:15 "server time" in the spec. Exness server time is UTC+3, and
    # Stage 2 measured the spread spike at 21:00 UTC, so that is the window used.
    rollover_start_min: int = 20 * 60 + 55
    rollover_end_min: int = 21 * 60 + 15

    max_gap_s: float = 60.0
    slippage_usd: float = 0.12

    risk_pct: float = 0.005
    daily_stop_pct: float = 0.02
    starting_equity: float = 100_000.0

    def __post_init__(self) -> None:
        if self.extend_rule not in ("cancel", "reset"):
            raise ValueError(f"extend_rule must be cancel/reset, got {self.extend_rule!r}")


def compute_signals(ticks: pl.DataFrame, params: LVFParams) -> pl.DataFrame:
    """Add V, E, rho and the setup mask. Vectorized - this is the hot path.

    Every column here depends only on the current tick and the ones before it.
    """
    n = params.window
    lagged = pl.col("ts").shift(n)

    frame = ticks.with_columns(
        mid=(pl.col("bid") + pl.col("ask")) / 2.0,
        spread=pl.col("ask") - pl.col("bid"),
    ).with_columns(
        move_signed=pl.col("mid") - pl.col("mid").shift(n),
        path=pl.col("mid").diff().abs().rolling_sum(n),
        window_s=(pl.col("ts") - lagged).dt.total_microseconds() / US,
        gap_s=(pl.col("ts") - pl.col("ts").shift(1)).dt.total_microseconds() / US,
    )

    frame = frame.with_columns(
        move=pl.col("move_signed").abs(),
        # A window containing any large inter-tick gap straddles a session
        # boundary; Move and Rate are both meaningless across it.
        window_has_gap=(pl.col("gap_s") > params.max_gap_s)
        .cast(pl.UInt32)
        .rolling_sum(n)
        > 0,
    )

    frame = frame.with_columns(
        rate=n / pl.col("window_s").clip(lower_bound=1e-3),
        efficiency=pl.when(pl.col("path") > 0)
        .then(pl.col("move") / pl.col("path"))
        .otherwise(None),
    )

    # Lagged baselines: judge a tick against what came before it, not including it.
    frame = frame.with_columns(
        b_move=pl.col("move").ewm_mean(span=params.ema_span, ignore_nulls=True).shift(1),
        b_rate=pl.col("rate").ewm_mean(span=params.ema_span, ignore_nulls=True).shift(1),
    )

    # dt.hour() is Int8, and hour * 60 silently wraps past 127. The cast is not
    # cosmetic: without it the rollover filter never fires.
    minute_of_day = pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute()
    frame = frame.with_columns(
        v=pl.when(pl.col("b_move") > 0).then(pl.col("move") / pl.col("b_move")).otherwise(None),
        rho=pl.when(pl.col("b_rate") > 0).then(pl.col("rate") / pl.col("b_rate")).otherwise(None),
        in_rollover=(minute_of_day >= params.rollover_start_min)
        & (minute_of_day < params.rollover_end_min),
    )

    return frame.with_columns(
        setup=(
            (pl.col("v") >= params.v_min)
            & (pl.col("efficiency") >= params.e_min)
            & (pl.col("rho") <= params.rho_max)
            & (pl.col("move") >= params.move_min)
            & (pl.col("spread") <= params.spread_max)
            & ~pl.col("in_rollover")
            & ~pl.col("window_has_gap")
        ).fill_null(False)
    )


@dataclass
class _State:
    """What has to survive from one monthly chunk to the next."""

    account: Account
    vacuum_ts: list[int]
    vacuum_dir: list[int]
    warm: bool = False


def _simulate_chunk(
    frame: pl.DataFrame,
    params: LVFParams,
    state: _State,
    spec,
    first_own: int,
    last_own: int,
) -> tuple[list[Trade], dict]:
    """Walk the candidate setups in one chunk and produce closed trades.

    ``first_own``/``last_own`` bound the indices this chunk is responsible for;
    the ticks outside them are overlap, present so that baselines are warm at the
    start and open positions can be managed to their exit at the end.
    """
    ts = frame["ts"].dt.timestamp("us").to_numpy()
    bid = frame["bid"].to_numpy()
    ask = frame["ask"].to_numpy()
    mid = frame["mid"].to_numpy()
    move_arr = frame["move"].to_numpy()
    move_signed = frame["move_signed"].to_numpy()
    setup = frame["setup"].to_numpy()
    v_arr = frame["v"].to_numpy()
    e_arr = frame["efficiency"].to_numpy()
    rho_arr = frame["rho"].to_numpy()

    total = len(ts)
    candidates = np.flatnonzero(setup)
    candidates = candidates[(candidates >= first_own) & (candidates <= last_own)]

    trades: list[Trade] = []
    counts = {
        "candidates": int(len(candidates)),
        "armed": 0,
        "cancelled_extension": 0,
        "skipped_cluster": 0,
        "skipped_daily_stop": 0,
        "skipped_size": 0,
        "entries": 0,
        "unresolved": 0,
    }

    slip = params.slippage_usd
    cluster_us = params.cluster_window_s * US
    next_free = first_own

    for i in candidates:
        if i < next_free:
            continue

        day = int(ts[i] // (86_400 * US))
        state.account.roll_to(day)

        direction_of_move = 1 if move_signed[i] > 0 else -1

        # "Not the 3rd vacuum in the same direction within 30 min."
        cutoff = ts[i] - cluster_us
        recent = sum(
            1
            for k in range(len(state.vacuum_ts) - 1, -1, -1)
            if state.vacuum_ts[k] >= cutoff and state.vacuum_dir[k] == direction_of_move
        )
        if recent >= params.cluster_max:
            counts["skipped_cluster"] += 1
            continue

        state.vacuum_ts.append(int(ts[i]))
        state.vacuum_dir.append(direction_of_move)
        if len(state.vacuum_ts) > 512:  # bounded history
            del state.vacuum_ts[:256]
            del state.vacuum_dir[:256]

        counts["armed"] += 1
        if not state.account.can_trade():
            counts["skipped_daily_stop"] += 1
            next_free = i + 1
            continue

        # --- arming: wait for the move to stall -----------------------------
        origin = mid[i - params.window]
        extreme = mid[i]
        setup_move = move_arr[i]
        held = 0
        entry_idx = -1
        j = i + 1
        scanned = 0

        while j < total and scanned < params.arm_max_ticks:
            scanned += 1
            extended = (
                mid[j] > extreme + params.extend_usd
                if direction_of_move > 0
                else mid[j] < extreme - params.extend_usd
            )
            if extended:
                if params.extend_rule == "cancel":
                    entry_idx = -2  # cancelled
                    break
                extreme = mid[j]
                held = 0
            else:
                held += 1
                if held >= params.arm_ticks:
                    entry_idx = j
                    break
            j += 1

        if entry_idx < 0:
            if entry_idx == -2:
                counts["cancelled_extension"] += 1
            next_free = min(j + 1, total)
            continue

        # --- entry ----------------------------------------------------------
        side = -direction_of_move  # fade it
        stop_distance = max(params.stop_frac * setup_move, params.stop_floor)
        if side < 0:  # short: sell at the bid, slipped down
            entry_price = bid[entry_idx] - slip
            stop_price = extreme + stop_distance
        else:  # long: buy at the ask, slipped up
            entry_price = ask[entry_idx] + slip
            stop_price = extreme - stop_distance

        lots = lots_for_risk(
            state.account.equity,
            params.risk_pct,
            entry_price,
            stop_price,
            spec.contract_size,
        )
        if lots <= 0:
            counts["skipped_size"] += 1
            next_free = entry_idx + 1
            continue

        leg = abs(extreme - origin)
        if side < 0:
            tp1 = extreme - params.tp1_frac * leg
            tp2 = extreme - params.tp2_frac * leg
        else:
            tp1 = extreme + params.tp1_frac * leg
            tp2 = extreme + params.tp2_frac * leg

        trade = Trade(
            symbol=spec.name,
            direction=side,
            entry=Fill(
                ts=int(ts[entry_idx]),
                price=entry_price,
                lots=lots,
                reason="entry",
                mid=float(mid[entry_idx]),
            ),
            stop_price=stop_price,
            contract_size=spec.contract_size,
            commission_per_lot_side=spec.commission_per_lot_side_usd,
            setup={
                "v": float(v_arr[i]),
                "efficiency": float(e_arr[i]),
                "rho": float(rho_arr[i]),
                "setup_move": float(setup_move),
                "leg": float(leg),
                "arm_ticks_used": int(scanned),
                "extreme": float(extreme),
            },
        )
        counts["entries"] += 1

        # --- management -------------------------------------------------------
        deadline = ts[entry_idx] + params.time_stop_s * US
        drift_deadline = ts[entry_idx] + 250_000  # 250 ms, for the slippage check
        drift_measured = None
        open_lots = lots
        current_stop = stop_price
        took_tp1 = False
        mae = 0.0
        mfe = 0.0

        k = entry_idx + 1
        while k < total and open_lots > 0:
            if drift_measured is None and ts[k] >= drift_deadline:
                # Adverse mid movement over a 250 ms fill latency, signed against
                # the position - the quantity the spec worries about.
                drift_measured = float(-side * (mid[k] - mid[entry_idx]))

            close_px = ask[k] if side < 0 else bid[k]
            excursion = side * (close_px - entry_price)
            mfe = max(mfe, excursion)
            mae = min(mae, excursion)

            stop_hit = close_px >= current_stop if side < 0 else close_px <= current_stop
            if stop_hit:
                # A stop is a market order: fill at the prevailing quote if it has
                # already jumped past the level, and take slippage on top.
                fill = (
                    max(current_stop, close_px) + slip
                    if side < 0
                    else min(current_stop, close_px) - slip
                )
                trade.exits.append(
                    Fill(int(ts[k]), fill, open_lots, "breakeven" if took_tp1 else "stop", float(mid[k]))
                )
                trade.slippage_usd += slip * open_lots * spec.contract_size
                open_lots = 0.0
                break

            if not took_tp1:
                reached_tp1 = close_px <= tp1 if side < 0 else close_px >= tp1
                if reached_tp1:
                    half = round(open_lots / 2, 2)
                    if half > 0:
                        # Limit order: fills at its price, never better.
                        trade.exits.append(Fill(int(ts[k]), tp1, half, "tp1", float(mid[k])))
                        open_lots = round(open_lots - half, 2)
                    took_tp1 = True
                    current_stop = entry_price  # stop to breakeven
                    if open_lots <= 0:
                        break

            if took_tp1:
                reached_tp2 = close_px <= tp2 if side < 0 else close_px >= tp2
                if reached_tp2:
                    trade.exits.append(Fill(int(ts[k]), tp2, open_lots, "tp2", float(mid[k])))
                    open_lots = 0.0
                    break

            if ts[k] >= deadline:
                fill = close_px + slip if side < 0 else close_px - slip
                trade.exits.append(Fill(int(ts[k]), fill, open_lots, "time", float(mid[k])))
                trade.slippage_usd += slip * open_lots * spec.contract_size
                open_lots = 0.0
                break
            k += 1

        if open_lots > 0:
            # Ran out of ticks inside this chunk's overlap. Should not happen with
            # a tail long enough to cover the time stop; counted so it cannot hide.
            last = min(k, total - 1)
            close_px = ask[last] if side < 0 else bid[last]
            trade.exits.append(Fill(int(ts[last]), close_px, open_lots, "truncated", float(mid[last])))
            counts["unresolved"] += 1

        trade.mae_price = mae
        trade.mfe_price = mfe
        trade.setup["drift_250ms_usd"] = drift_measured
        trade.slippage_usd += slip * lots * spec.contract_size  # entry
        state.account.apply(trade.net_usd)
        trades.append(trade)
        next_free = min(k + 1, total)

    return trades, counts


def _month_range(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def backtest(
    symbol: str,
    start: date,
    end: date,
    params: LVFParams | None = None,
    *,
    head_ticks: int = 120_000,
    tail_ticks: int = 200_000,
    progress: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Run LVF over a date range, month by month, carrying account state.

    Each chunk is loaded with a head of previous-month ticks so the EMA
    baselines are warm at its first candidate, and a tail of next-month ticks so
    a position opened near the boundary can be managed to its real exit rather
    than force-closed at midnight.
    """
    params = params or LVFParams()
    spec = get_spec(symbol)
    state = _State(
        account=Account(
            equity=params.starting_equity,
            risk_pct=params.risk_pct,
            daily_stop_pct=params.daily_stop_pct,
        ),
        vacuum_ts=[],
        vacuum_dir=[],
    )

    months = _month_range(start, end)
    all_trades: list[dict] = []
    totals = {
        "candidates": 0,
        "armed": 0,
        "cancelled_extension": 0,
        "skipped_cluster": 0,
        "skipped_daily_stop": 0,
        "skipped_size": 0,
        "entries": 0,
        "unresolved": 0,
        "ticks": 0,
    }

    for index, (year, month) in enumerate(months):
        own_path = paths.tick_parquet_path(spec.name, year, month)
        if not own_path.exists():
            continue

        pieces = []
        if index > 0:
            prev = paths.tick_parquet_path(spec.name, *months[index - 1])
            if prev.exists():
                pieces.append(pl.read_parquet(prev).tail(head_ticks))
        own = pl.read_parquet(own_path)
        pieces.append(own)
        if index + 1 < len(months):
            nxt = paths.tick_parquet_path(spec.name, *months[index + 1])
            if nxt.exists():
                pieces.append(pl.read_parquet(nxt).head(tail_ticks))

        head_len = pieces[0].height if index > 0 and len(pieces) > 1 else 0
        combined = pl.concat(pieces) if len(pieces) > 1 else pieces[0]
        combined = clean_ticks(combined, DEFAULT_POLICY)

        # Cleaning can drop rows, so recover the boundaries by timestamp.
        own_lo, own_hi = own["ts"][0], own["ts"][-1]
        idx = combined["ts"]
        first_own = int((idx < own_lo).sum())
        last_own = int((idx <= own_hi).sum()) - 1

        frame = compute_signals(combined, params)
        trades, counts = _simulate_chunk(
            frame, params, state, spec, first_own, last_own
        )
        all_trades.extend(t.to_dict() for t in trades)
        for key, value in counts.items():
            totals[key] += value
        totals["ticks"] += own.height

        if progress:
            print(
                f"  {year}-{month:02d}  ticks {own.height:>9,}  "
                f"cand {counts['candidates']:>6,}  entries {counts['entries']:>4,}  "
                f"equity ${state.account.equity:,.0f}",
                flush=True,
            )
        del combined, frame, own, pieces

    totals["final_equity"] = state.account.equity
    totals["blocked_days"] = state.account.blocked_days
    schema_frame = pl.DataFrame(all_trades) if all_trades else pl.DataFrame()
    return schema_frame, totals


def forward_study(
    symbol: str,
    start: date,
    end: date,
    params: LVFParams | None = None,
    *,
    horizons_s: tuple[int, ...] = (60, 300, 900, 2700),
    drop_rho_filter: bool = True,
    head_ticks: int = 120_000,
    tail_ticks: int = 400_000,
) -> pl.DataFrame:
    """Measure the hypothesis directly, with no trade construction in the way.

    For every setup, record how far the mid moves afterwards *in the fade
    direction*, at several horizons. This separates two very different failures:
    a hypothesis that is simply wrong, and a hypothesis that is right but whose
    entry and exit rules give the edge away.

    ``drop_rho_filter`` keeps the participation test out of the mask so that
    setups can be bucketed by rho afterwards - which is the only way to check the
    strategy's central claim, that low participation is what makes a move revert.
    """
    params = params or LVFParams()
    spec = get_spec(symbol)
    rows: list[dict] = []

    months = _month_range(start, end)
    for index, (year, month) in enumerate(months):
        own_path = paths.tick_parquet_path(spec.name, year, month)
        if not own_path.exists():
            continue
        pieces = []
        if index > 0:
            prev = paths.tick_parquet_path(spec.name, *months[index - 1])
            if prev.exists():
                pieces.append(pl.read_parquet(prev).tail(head_ticks))
        own = pl.read_parquet(own_path)
        pieces.append(own)
        if index + 1 < len(months):
            nxt = paths.tick_parquet_path(spec.name, *months[index + 1])
            if nxt.exists():
                pieces.append(pl.read_parquet(nxt).head(tail_ticks))

        combined = clean_ticks(pl.concat(pieces) if len(pieces) > 1 else pieces[0], DEFAULT_POLICY)
        frame = compute_signals(combined, params)

        mask = (
            (pl.col("v") >= params.v_min)
            & (pl.col("efficiency") >= params.e_min)
            & (pl.col("move") >= params.move_min)
            & (pl.col("spread") <= params.spread_max)
            & ~pl.col("in_rollover")
            & ~pl.col("window_has_gap")
        )
        if not drop_rho_filter:
            mask = mask & (pl.col("rho") <= params.rho_max)
        frame = frame.with_columns(raw_setup=mask.fill_null(False))

        ts = frame["ts"].dt.timestamp("us").to_numpy()
        mid = frame["mid"].to_numpy()
        own_lo, own_hi = own["ts"][0], own["ts"][-1]
        idx = frame["ts"]
        first_own = int((idx < own_lo).sum())
        last_own = int((idx <= own_hi).sum()) - 1

        setup_idx = np.flatnonzero(frame["raw_setup"].to_numpy())
        setup_idx = setup_idx[(setup_idx >= first_own) & (setup_idx <= last_own)]
        if len(setup_idx) == 0:
            continue

        # One observation per cluster: consecutive qualifying ticks are the same
        # event seen repeatedly, and counting them all would weight long clusters.
        keep = [setup_idx[0]]
        for candidate in setup_idx[1:]:
            if ts[candidate] - ts[keep[-1]] > 60 * US:
                keep.append(candidate)
        keep_arr = np.array(keep)

        signed = frame["move_signed"].to_numpy()[keep_arr]
        fade = -np.sign(signed)  # the direction the strategy would trade
        record = {
            "ts": ts[keep_arr],
            "v": frame["v"].to_numpy()[keep_arr],
            "rho": frame["rho"].to_numpy()[keep_arr],
            "efficiency": frame["efficiency"].to_numpy()[keep_arr],
            "move": frame["move"].to_numpy()[keep_arr],
            "fade_dir": fade,
        }
        for horizon in horizons_s:
            target = np.searchsorted(ts, ts[keep_arr] + horizon * US, side="left")
            target = np.clip(target, 0, len(ts) - 1)
            delta = mid[target] - mid[keep_arr]
            record[f"fade_{horizon}s_usd"] = fade * delta
            # Scaled by the move that triggered the setup: "how much of the
            # vacuum came back", which is comparable across price levels.
            record[f"fade_{horizon}s_frac"] = fade * delta / record["move"]
        rows.append(pl.DataFrame(record))
        del combined, frame, own, pieces

    return pl.concat(rows) if rows else pl.DataFrame()
