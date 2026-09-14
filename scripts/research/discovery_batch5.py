"""Discovery batch 5: the H14 conditional micro-structure screen, pre-registered.

H14 exactly as registered in docs/findings/discovery-program.md, which was
written before this script was first run. A grid of conditional cells - event
family x parameters x condition x horizon x side - screened on half A of dev
(odd months) under Benjamini-Hochberg, confirmed on half B (even months), and
the whole procedure re-run on date-shifted placebo outcomes to count how many
survivors chance produces. Survivors become one portfolio, which is the only
unit the validation split has the power to confirm.

    python scripts/research/discovery_batch5.py dev [--placebos 3]
    python scripts/research/discovery_batch5.py validate

``dev`` never reads validation or test. ``validate`` computes the expected t
from event timestamps first and reads outcomes only if the registered power
rule is met.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qlab.costs import CostModel, SlippageModel
from qlab.loader import load_bars
from qlab.stats import benjamini_hochberg, normal_sf
from qlab.symbols import get_spec

SYMBOLS = ("EURUSD", "USDJPY", "XAUUSD", "USTEC")
HORIZONS = (1, 3, 5, 10, 20, 60)
OUT = Path("reports/discovery/batch5")
SIGMA_WINDOW = 1440
VR_WINDOW = 20 * 1440

LEVELS = (30, 60, 240, "PD")
KS = (1, 3, 5)
DEPTH_EDGES = np.array([0.0, 0.15, 0.40, 1.0])
XM_GRID = ((5, 2.5), (5, 3.5), (15, 2.5), (15, 3.5))
AR_WINDOWS = (5, 15)
OD_ANCHORS = (("Tokyo", "Asia/Tokyo", 9 * 60), ("London", "Europe/London", 8 * 60),
              ("NewYork", "America/New_York", 9 * 60 + 30))
OD_MS = (5, 15, 30)
CONT_FAMILIES = {"BO", "OD"}

Q_A = 0.10
T_B = 1.96
MIN_N = 100
MIN_DAYS = 60
POWER_T = 2.5
JPY_REF = 150.0  # validation-era USDJPY, only to put commission into pips
MINUTE_US = 60_000_000

TW_NAMES = ("Asia", "London", "NY-AM", "NY-PM")
VR_NAMES = ("vol-low", "vol-mid", "vol-high")
USD_NAMES = ("usd-aligned", "usd-opposed", "usd-flat")
TR_NAMES = ("trend-with", "trend-against", "trend-flat")


# --- preparation ----------------------------------------------------------


def diurnal_table(bars: pl.DataFrame) -> pl.DataFrame:
    overall = bars.select((pl.col("r1") ** 2).mean()).item()
    return bars.group_by("bucket").agg(d=((pl.col("r1") ** 2).mean() / overall).sqrt())


def move_z(bars: pl.DataFrame, window: int) -> pl.DataFrame:
    """``r{W}`` and ``z{W}`` exactly as batch 1: contiguous, sigma from before the window."""
    contiguous = (pl.col("ts") - pl.col("ts").shift(window)).dt.total_minutes() == window
    r = pl.when(contiguous).then(pl.col("lc") - pl.col("lc").shift(window))
    expected = pl.col("sig").shift(window) * pl.col("d2").rolling_sum(window).sqrt()
    return bars.with_columns(r.alias(f"r{window}")).with_columns(
        (pl.col(f"r{window}") / expected).alias(f"z{window}")
    )


def prepare(symbol: str, split: str, diurnal: pl.DataFrame | None):
    warm = None if split == "dev" else timedelta(days=45)
    bars = load_bars(symbol, "1m", split=split, warmup=warm,
                     allow_test=(split == "test")).sort("ts")
    if "is_warmup" not in bars.columns:
        bars = bars.with_columns(is_warmup=pl.lit(False))
    bars = bars.with_columns(
        lc=pl.col("close").log(),
        _gap=(pl.col("ts") - pl.col("ts").shift(1)).dt.total_minutes(),
        bucket=pl.col("ts_open").dt.hour().cast(pl.Int32) * 4
        + pl.col("ts_open").dt.minute().cast(pl.Int32) // 15,
        hour=pl.col("ts_open").dt.hour().cast(pl.Int32),
    ).with_columns(
        r1=pl.when(pl.col("_gap") == 1).then(pl.col("lc") - pl.col("lc").shift(1)),
    )
    if diurnal is None:
        diurnal = diurnal_table(bars.filter(~pl.col("is_warmup")))
    ny = pl.col("ts_open").dt.convert_time_zone("America/New_York")
    bars = (
        bars.join(diurnal, on="bucket", how="left")
        .sort("ts")
        .with_columns(
            sig=(pl.col("r1") / pl.col("d")).rolling_std(
                window_size=SIGMA_WINDOW, min_samples=SIGMA_WINDOW // 2),
            d2=pl.col("d") ** 2,
            ny_min=(ny.dt.hour().cast(pl.Int32) * 60 + ny.dt.minute().cast(pl.Int32)),
            tday=(ny + pl.duration(hours=7)).dt.date(),
            _gap_ts=pl.when(pl.col("_gap").is_null() | (pl.col("_gap") >= 30))
            .then(pl.col("ts_open")).forward_fill(),
        )
        .with_columns(
            vr=pl.col("sig") / pl.col("sig").rolling_mean(VR_WINDOW, min_samples=VR_WINDOW // 4),
            post_gap=(pl.col("ts") - pl.col("_gap_ts")).dt.total_minutes() <= 60,
        )
    )
    for w in (5, 15, 30):
        bars = move_z(bars, w)

    # Trading-day-to-date move, z-scored by the diurnal variance accumulated so far.
    bars = bars.with_columns(
        tr_z=(pl.col("lc") - pl.col("lc").first().over("tday"))
        / (pl.col("sig") * pl.col("d2").cum_sum().over("tday").sqrt()),
        tdrank=pl.col("tday").rank("dense").cast(pl.Int64),
    )

    # ATR30: mean range of the prior 48 completed 30-minute bars.
    thirty = (
        bars.group_by_dynamic("ts_open", every="30m", closed="left", label="left")
        .agg(hi=pl.col("high").max(), lo=pl.col("low").min())
        .with_columns(_end=pl.col("ts_open") + pl.duration(minutes=30))
        .with_columns(atr=(pl.col("hi") - pl.col("lo")).rolling_mean(48, min_samples=24))
        .select("_end", "atr")
        .sort("_end")
    )
    bars = bars.join_asof(thirty, left_on="ts", right_on="_end", strategy="backward")

    daily = (
        bars.group_by("tday").agg(dh=pl.col("high").max(), dl=pl.col("low").min())
        .sort("tday")
        .with_columns(pdh=pl.col("dh").shift(1), pdl=pl.col("dl").shift(1))
        .select("tday", "pdh", "pdl")
    )
    bars = bars.join(daily, on="tday", how="left").sort("ts")
    level_cols = []
    for n in (30, 60, 240):
        bars = bars.with_columns(
            pl.col("high").rolling_max(n).shift(1).alias(f"hi{n}"),
            pl.col("low").rolling_min(n).shift(1).alias(f"lo{n}"),
        )
        level_cols += [f"hi{n}", f"lo{n}"]

    fwd = []
    for h in HORIZONS:
        ok = (pl.col("ts").shift(-h) - pl.col("ts")).dt.total_minutes() == h
        fwd.append(pl.when(ok).then((pl.col("close").shift(-h) / pl.col("close") - 1) * 1e4)
                   .alias(f"f{h}"))
    bars = bars.with_columns(fwd)

    m = pl.col("ny_min")
    tw = (
        pl.when((m >= 18 * 60 + 30) | (m < 120)).then(0)
        .when((m >= 120) & (m < 480)).then(1)
        .when((m >= 480) & (m < 690)).then(2)
        .when((m >= 690) & (m < 990)).then(3)
        .otherwise(-1)
    )
    bars = bars.with_columns(
        tw=tw.cast(pl.Int8),
        excluded=((m >= 16 * 60 + 30) & (m < 18 * 60 + 30))
        | pl.col("post_gap").fill_null(True)
        | pl.col("is_warmup"),
        half=(pl.col("ts").dt.month() % 2 == 0).cast(pl.Int8),  # 0 = A (odd), 1 = B (even)
    )
    return bars, diurnal


def rt_by_hour(symbol: str, split: str, adverse: float) -> np.ndarray:
    """Round turn in pips per UTC hour, measured on ``split``."""
    cost = CostModel.from_profiles(symbol, split=split,
                                   slippage=SlippageModel(adverse_fraction=adverse))
    price = JPY_REF if get_spec(symbol).quote_ccy != "USD" else 1.0
    table = cost.hourly_table(price)
    return table.sort("hour")["round_turn_pips"].to_numpy()


def cross_features(prepared: dict[str, pl.DataFrame], betas: dict | None):
    """USD direction implied for each target by the *other* instruments, and DX residuals."""
    z = None
    for s in SYMBOLS:
        f = prepared[s].select("ts", pl.col("z30").alias(f"z30_{s}"), pl.col("r15").alias(f"r15_{s}"),
                               pl.col("half").alias(f"half_{s}"))
        z = f if z is None else z.join(f, on="ts", how="full", coalesce=True)
    z = z.sort("ts")
    implied = {
        "EURUSD": -pl.col("z30_USDJPY"),
        "USDJPY": -pl.col("z30_EURUSD"),
        "XAUUSD": -(pl.col("z30_USDJPY") - pl.col("z30_EURUSD")) / math.sqrt(2),
        "USTEC": -(pl.col("z30_USDJPY") - pl.col("z30_EURUSD")) / math.sqrt(2),
    }
    leg = {
        "EURUSD": pl.col("r15_USDJPY"),
        "USDJPY": pl.col("r15_EURUSD"),
        "XAUUSD": (pl.col("r15_USDJPY") - pl.col("r15_EURUSD")) / 2,
    }
    fit_betas = betas is None
    betas = dict(betas or {})
    out = {}
    for s in SYMBOLS:
        cols = [implied[s].alias("usd_z")]
        frame = z.with_columns(cols)
        if s in leg:
            frame = frame.with_columns(_x=leg[s], _y=pl.col(f"r15_{s}"))
            if fit_betas:
                g = frame.filter((pl.col("ts").dt.minute() % 15 == 0) & (pl.col(f"half_{s}") == 0)) \
                    .select("_x", "_y").drop_nulls()
                x, y = g["_x"].to_numpy(), g["_y"].to_numpy()
                x, y = x - x.mean(), y - y.mean()
                betas[s] = float((x * y).sum() / (x * x).sum())
            frame = frame.with_columns(_e=pl.col("_y") - betas[s] * pl.col("_x")).with_columns(
                dx=pl.col("_e") / pl.col("_e").rolling_std(1440, min_samples=720).shift(15)
            )
            keep = ["ts", "usd_z", "dx"]
        else:
            keep = ["ts", "usd_z"]
        out[s] = prepared[s].join(frame.select(keep), on="ts", how="left").sort("ts")
    return out, betas


@dataclass
class Feats:
    symbol: str
    ts: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    fwd: np.ndarray
    excluded: np.ndarray
    tw: np.ndarray
    vr: np.ndarray
    usd_z: np.ndarray
    tr_z: np.ndarray
    half: np.ndarray
    tdrank: np.ndarray
    ny_min: np.ndarray
    hour: np.ndarray
    atr: np.ndarray
    rt_fade: np.ndarray = field(default=None)
    rt_chase: np.ndarray = field(default=None)
    _key_sorted: np.ndarray = field(default=None)
    _key_order: np.ndarray = field(default=None)


def to_feats(symbol: str, bars: pl.DataFrame, split_cost: str) -> Feats:
    a = lambda c, dt=np.float64: bars[c].fill_null(np.nan).to_numpy().astype(dt) if dt == np.float64 \
        else bars[c].to_numpy().astype(dt)
    spec = get_spec(symbol)
    f = Feats(
        symbol=symbol,
        ts=bars["ts"].dt.epoch("us").to_numpy(),
        high=a("high"), low=a("low"), close=a("close"),
        fwd=np.column_stack([a(f"f{h}") for h in HORIZONS]),
        excluded=bars["excluded"].to_numpy().astype(bool),
        tw=a("tw", np.int8), vr=a("vr"), usd_z=a("usd_z"), tr_z=a("tr_z"),
        half=a("half", np.int8), tdrank=a("tdrank", np.int64), ny_min=a("ny_min", np.int64),
        hour=a("hour", np.int64), atr=a("atr"),
    )
    to_bps = spec.pip / f.close * 1e4
    f.rt_fade = rt_by_hour(symbol, split_cost, 0.5)[f.hour] * to_bps
    f.rt_chase = rt_by_hour(symbol, split_cost, 1.0)[f.hour] * to_bps
    key = f.tdrank * 1440 + f.ny_min
    f._key_order = np.argsort(key, kind="stable")
    f._key_sorted = key[f._key_order]
    return f


# --- events ---------------------------------------------------------------


@dataclass
class Base:
    symbol: str
    family: str
    param: str
    idx: np.ndarray
    direction: np.ndarray

    @property
    def cont(self) -> bool:
        return self.family in CONT_FAMILIES

    @property
    def name(self) -> str:
        return f"{self.symbol} {self.family} {self.param}"


def first_crossings(ts: np.ndarray, flag: np.ndarray, cooldown_min: int) -> np.ndarray:
    gap = cooldown_min * MINUTE_US
    keep, last = [], -(10**18)
    for i in np.flatnonzero(flag):
        if ts[i] - last >= gap:
            keep.append(i)
            last = ts[i]
    return np.asarray(keep, dtype=np.int64)


def sweep_bases(f: Feats, bars: pl.DataFrame) -> list[Base]:
    """SW and BO from the same breach episodes, for every level, K and depth bucket."""
    ts, high, low, close, atr = f.ts, f.high, f.low, f.close, f.atr
    n = ts.size
    acc: dict[tuple, tuple[list, list]] = {}
    for lvl in LEVELS:
        if lvl == "PD":
            his, los = bars["pdh"].fill_null(np.nan).to_numpy(), bars["pdl"].fill_null(np.nan).to_numpy()
        else:
            his, los = (bars[f"hi{lvl}"].fill_null(np.nan).to_numpy(),
                        bars[f"lo{lvl}"].fill_null(np.nan).to_numpy())
        lname = "PD" if lvl == "PD" else f"H{lvl}"
        for side in (1, -1):
            lev = his if side == 1 else los
            with np.errstate(invalid="ignore"):
                beyond = (high > lev) if side == 1 else (low < lev)
            beyond &= np.isfinite(lev)
            prev = np.concatenate([[False], beyond[:-1]])
            starts = np.flatnonzero(beyond & ~prev)
            last = -(10**18)
            exts = [0.0] * 5
            for i in starts:
                if ts[i] - last < 10 * MINUTE_US:
                    continue
                A = atr[i]
                if not A > 0:
                    continue
                last = ts[i]
                L = lev[i]
                ext, r, upto = 0.0, -1, -1
                for k in range(5):
                    j = i + k
                    if j >= n or ts[j] - ts[i] != k * MINUTE_US:
                        break
                    e = (high[j] - L) if side == 1 else (L - low[j])
                    if e > ext:
                        ext = e
                    exts[k] = ext
                    upto = k
                    inside = close[j] < L if side == 1 else close[j] > L
                    if inside:
                        r = k
                        break
                for K in KS:
                    if 0 <= r < K:
                        kind, j, dep, d = "SW", i + r, exts[r], -side
                    elif upto >= K - 1 and (r < 0 or r >= K):
                        kind, j, dep, d = "BO", i + K - 1, exts[K - 1], side
                    else:
                        continue
                    dep /= A
                    if dep > DEPTH_EDGES[-1]:
                        continue
                    b = int(np.searchsorted(DEPTH_EDGES, dep, side="right") - 1)
                    key = (kind, f"{lname} K{K} D{b + 1}")
                    lst = acc.setdefault(key, ([], []))
                    lst[0].append(j)
                    lst[1].append(d)
    bases = []
    for (kind, param), (idx, d) in sorted(acc.items()):
        bases.append(Base(f.symbol, kind, param, np.asarray(idx, np.int64), np.asarray(d, np.int8)))
    return bases


def other_bases(f: Feats, bars: pl.DataFrame) -> list[Base]:
    out = []
    for w, k in XM_GRID:
        z = bars[f"z{w}"].fill_null(np.nan).to_numpy()
        with np.errstate(invalid="ignore"):
            idx = first_crossings(f.ts, np.isfinite(z) & (np.abs(z) >= k), 15)
        out.append(Base(f.symbol, "XM", f"W{w} k{k}", idx, (-np.sign(z[idx])).astype(np.int8)))
    minute = bars["ts"].dt.minute().to_numpy()
    for w in AR_WINDOWS:
        z = bars[f"z{w}"].fill_null(np.nan).to_numpy()
        with np.errstate(invalid="ignore"):
            idx = np.flatnonzero((minute % 5 == 0) & np.isfinite(z) & (np.abs(z) >= 1.0))
        out.append(Base(f.symbol, "AR", f"W{w}", idx, (-np.sign(z[idx])).astype(np.int8)))
    # Opening drive: anchors in local time, DST-aware.
    stamp = {t: i for i, t in enumerate(f.ts)}
    for name, tz, anchor in OD_ANCHORS:
        local = bars["ts"].dt.convert_time_zone(tz)
        lmin = (local.dt.hour().cast(pl.Int64) * 60 + local.dt.minute().cast(pl.Int64)).to_numpy()
        a_idx = np.flatnonzero(lmin == anchor)
        for M in OD_MS:
            idx, d = [], []
            for a in a_idx:
                j = stamp.get(f.ts[a] + M * MINUTE_US)
                if j is None:
                    continue
                s = np.sign(f.close[j] - f.close[a])
                if s != 0:
                    idx.append(j)
                    d.append(int(s))
            out.append(Base(f.symbol, "OD", f"{name} M{M}", np.asarray(idx, np.int64),
                            np.asarray(d, np.int8)))
    if "dx" in bars.columns:
        dx = bars["dx"].fill_null(np.nan).to_numpy()
        with np.errstate(invalid="ignore"):
            idx = first_crossings(f.ts, np.isfinite(dx) & (np.abs(dx) >= 2.0), 15)
        out.append(Base(f.symbol, "DX", "e15 k2", idx, (-np.sign(dx[idx])).astype(np.int8)))
    return out


def clean_base(b: Base, f: Feats) -> Base:
    keep = ~f.excluded[b.idx] & (f.tw[b.idx] >= 0) & (b.direction != 0) & np.isfinite(f.rt_fade[b.idx])
    return Base(b.symbol, b.family, b.param, b.idx[keep], b.direction[keep])


# --- cells ----------------------------------------------------------------


def condition_masks(tw, vr, usd_rel, tr_rel):
    vr_c = np.full(vr.size, -1, np.int8)
    vr_c[vr < 0.8] = 0
    vr_c[(vr >= 0.8) & (vr <= 1.25)] = 1
    vr_c[vr > 1.25] = 2
    usd_c = np.full(usd_rel.size, -1, np.int8)
    usd_c[usd_rel > 0.5] = 0
    usd_c[usd_rel < -0.5] = 1
    usd_c[np.abs(usd_rel) <= 0.5] = 2
    tr_c = np.full(tr_rel.size, -1, np.int8)
    tr_c[tr_rel > 1] = 0
    tr_c[tr_rel < -1] = 1
    tr_c[np.abs(tr_rel) <= 1] = 2
    masks = [("all", np.ones(tw.size, bool))]
    for i, nm in enumerate(TW_NAMES):
        masks.append((nm, tw == i))
    for i, nm in enumerate(VR_NAMES):
        masks.append((nm, vr_c == i))
    for i, nm in enumerate(USD_NAMES):
        masks.append((nm, usd_c == i))
    for i, nm in enumerate(TR_NAMES):
        masks.append((nm, tr_c == i))
    for i, a in enumerate(TW_NAMES):
        for j, b in enumerate(VR_NAMES):
            masks.append((f"{a}&{b}", (tw == i) & (vr_c == j)))
    for i, a in enumerate(TW_NAMES):
        for j, b in enumerate(USD_NAMES):
            masks.append((f"{a}&{b}", (tw == i) & (usd_c == j)))
    return masks


def per_day_stats(day: np.ndarray, values: np.ndarray):
    """Per horizon: n events, event days, mean per event, per-day t."""
    H = values.shape[1]
    n = np.zeros(H, np.int64)
    days = np.zeros(H, np.int64)
    mean = np.full(H, np.nan)
    t = np.full(H, np.nan)
    if day.size == 0:
        return n, days, mean, t
    d0 = day - day.min()
    size = int(d0.max()) + 1
    for h in range(H):
        v = values[:, h]
        ok = np.isfinite(v)
        k = int(ok.sum())
        n[h] = k
        if k == 0:
            continue
        mean[h] = v[ok].mean()
        cnt = np.bincount(d0[ok], minlength=size)
        sums = np.bincount(d0[ok], weights=v[ok], minlength=size)[cnt > 0]
        days[h] = sums.size
        if sums.size >= 3:
            sd = sums.std(ddof=1)
            if sd > 0:
                t[h] = sums.mean() / sd * math.sqrt(sums.size)
    return n, days, mean, t


def outcome_index(b: Base, f: Feats, seed: int | None) -> np.ndarray:
    """Bar index whose forward return is this event's outcome; -1 if none.

    For a placebo, the same New York clock minute 1-5 trading days away.
    """
    if seed is None:
        return b.idx
    rng = np.random.default_rng(seed * 100_003 + hash(b.name) % 99_991)
    off = rng.integers(1, 6, size=b.idx.size) * rng.choice([-1, 1], size=b.idx.size)
    key = (f.tdrank[b.idx] + off) * 1440 + f.ny_min[b.idx]
    pos = np.searchsorted(f._key_sorted, key)
    pos = np.clip(pos, 0, f._key_sorted.size - 1)
    hit = f._key_sorted[pos] == key
    return np.where(hit, f._key_order[pos], -1)


def evaluate(bases: list[Base], feats: dict[str, Feats], seed: int | None = None) -> pl.DataFrame:
    rows = []
    for bi, b in enumerate(bases):
        f = feats[b.symbol]
        if b.idx.size < MIN_N:
            continue
        oi = outcome_index(b, f, seed)
        M = np.full((b.idx.size, len(HORIZONS)), np.nan)
        ok = oi >= 0
        M[ok] = f.fwd[oi[ok]]
        M *= b.direction[:, None]
        tw, vr, half, day = f.tw[b.idx], f.vr[b.idx], f.half[b.idx], f.tdrank[b.idx]
        usd = f.usd_z[b.idx] * b.direction
        tr = f.tr_z[b.idx] * b.direction
        for side in (1, -1):
            cont = b.cont if side == 1 else not b.cont
            rt = (f.rt_chase if cont else f.rt_fade)[b.idx]
            net = side * M - rt[:, None]
            for cname, cm in condition_masks(tw, vr, usd * side, tr * side):
                ma = cm & (half == 0)
                if ma.sum() < MIN_N:
                    continue
                nA, dA, mA, tA = per_day_stats(day[ma], net[ma])
                mb = cm & (half == 1)
                nB, dB, mB, tB = per_day_stats(day[mb], net[mb])
                midA = np.nanmean(side * M[ma], axis=0)
                for h, hz in enumerate(HORIZONS):
                    if nA[h] < MIN_N or dA[h] < MIN_DAYS:
                        continue
                    rows.append((bi, b.symbol, b.family, b.param, cname, hz, side,
                                 int(nA[h]), int(dA[h]), float(midA[h]), float(mA[h]), float(tA[h]),
                                 int(nB[h]), int(dB[h]), float(mB[h]), float(tB[h])))
    table = pl.DataFrame(rows, schema=["base", "symbol", "family", "param", "cond", "h", "side",
                                       "nA", "daysA", "midA", "netA", "tA",
                                       "nB", "daysB", "netB", "tB"], orient="row")
    p = np.array([normal_sf(t) if math.isfinite(t) else 1.0 for t in table["tA"].to_numpy()])
    q = benjamini_hochberg(p)
    return table.with_columns(pA=pl.Series(p), qA=pl.Series(q)).with_columns(
        passA=(pl.col("qA") <= Q_A) & (pl.col("netA") > 0),
    ).with_columns(
        passB=pl.col("passA") & (pl.col("netB") > 0) & (pl.col("tB") >= T_B),
    )


# --- stage C --------------------------------------------------------------


def cell_events(row: dict, bases: list[Base], feats: dict[str, Feats]):
    """Events of one cell: (bar idx, traded direction, outcome net_now at its horizon)."""
    b = bases[row["base"]]
    f = feats[b.symbol]
    side = row["side"]
    usd = f.usd_z[b.idx] * b.direction * side
    tr = f.tr_z[b.idx] * b.direction * side
    masks = dict(condition_masks(f.tw[b.idx], f.vr[b.idx], usd, tr))
    m = masks[row["cond"]]
    h = HORIZONS.index(row["h"])
    cont = b.cont if side == 1 else not b.cont
    rt = (f.rt_chase if cont else f.rt_fade)[b.idx]
    net = side * b.direction * f.fwd[b.idx, h] - rt
    return b.idx[m], (side * b.direction)[m], net[m]


def dedupe(surv: pl.DataFrame, bases, feats) -> list[dict]:
    kept: list[dict] = []
    kept_ts: dict[str, np.ndarray] = {}
    for row in surv.sort("tA", descending=True).iter_rows(named=True):
        idx, _, _ = cell_events(row, bases, feats)
        f = feats[row["symbol"]]
        ts = np.sort(f.ts[idx])
        prior = kept_ts.get(row["symbol"])
        if prior is not None and prior.size:
            pos = np.searchsorted(prior, ts)
            lo = np.abs(ts - prior[np.clip(pos - 1, 0, prior.size - 1)])
            hi = np.abs(prior[np.clip(pos, 0, prior.size - 1)] - ts)
            near = (np.minimum(lo, hi) <= 5 * MINUTE_US).mean()
            if near >= 0.5:
                continue
        kept.append(row)
        kept_ts[row["symbol"]] = np.sort(np.concatenate([prior, ts])) if prior is not None else ts
    return kept


def portfolio(kept: list[dict], bases, feats, half: int | None):
    """Daily sums of net_now across every kept cell's trades (one unit each)."""
    frames = []
    for row in kept:
        idx, _, net = cell_events(row, bases, feats)
        f = feats[row["symbol"]]
        m = np.isfinite(net) & ((f.half[idx] == half) if half is not None else True)
        frames.append(pl.DataFrame({"tday": f.tdrank[idx][m], "net": net[m]}))
    if not frames:
        return {}
    tape = pl.concat(frames)
    daily = tape.group_by("tday").agg(pl.col("net").sum())["net"].to_numpy()
    t = daily.mean() / daily.std(ddof=1) * math.sqrt(daily.size) if daily.size > 2 else float("nan")
    return {"trades": tape.height, "days": int(daily.size), "net_per_trade": float(tape["net"].mean()),
            "bps_per_day": float(daily.mean()), "t_day": float(t),
            "sharpe_ann": float(daily.mean() / daily.std(ddof=1) * math.sqrt(252))}


# --- driver ---------------------------------------------------------------


def build(split: str, state: dict | None):
    prepared, diurnals = {}, {}
    for s in SYMBOLS:
        t0 = time.time()
        d = None if state is None else pl.DataFrame(state["diurnal"][s])
        prepared[s], diurnals[s] = prepare(s, split, d)
        print(f"  prepared {s} {prepared[s].height:,} bars ({time.time() - t0:.0f}s)", flush=True)
    joined, betas = cross_features(prepared, None if state is None else state["betas"])
    del prepared
    feats, bases = {}, []
    cost_split = "validation" if split == "dev" else split
    for s in SYMBOLS:
        f = to_feats(s, joined[s], cost_split)
        feats[s] = f
        raw = sweep_bases(f, joined[s]) + other_bases(f, joined[s])
        bases += [clean_base(b, f) for b in raw]
        joined[s] = None
        print(f"  events {s}: {sum(b.idx.size for b in bases if b.symbol == s):,}", flush=True)
    return feats, bases, diurnals, betas


def dev(placebos: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats, bases, diurnals, betas = build("dev", None)
    print(f"{len(bases)} base event series", flush=True)

    t0 = time.time()
    table = evaluate(bases, feats)
    table.write_parquet(OUT / "cells.parquet")
    nA, nB = int(table["passA"].sum()), int(table["passB"].sum())
    print(f"real: {table.height:,} eligible cells, stage A {nA}, stage B {nB} ({time.time() - t0:.0f}s)",
          flush=True)

    plac = []
    for seed in range(1, placebos + 1):
        pt = evaluate(bases, feats, seed=seed)
        plac.append({"seed": seed, "cells": pt.height, "passA": int(pt["passA"].sum()),
                     "passB": int(pt["passB"].sum())})
        pt.filter(pl.col("passA")).write_parquet(OUT / f"placebo{seed}_passA.parquet")
        print(f"placebo {seed}: stage A {plac[-1]['passA']}, stage B {plac[-1]['passB']}", flush=True)
    mean_plac_B = float(np.mean([p["passB"] for p in plac])) if plac else float("nan")
    informative = nB > 0 and (not plac or mean_plac_B < 0.5 * nB)

    surv = table.filter(pl.col("passB"))
    kept = dedupe(surv, bases, feats) if surv.height else []
    port_B = portfolio(kept, bases, feats, half=1) if kept else {}
    port_A = portfolio(kept, bases, feats, half=0) if kept else {}

    state = {
        "diurnal": {s: diurnals[s].to_dict(as_series=False) for s in SYMBOLS},
        "betas": betas,
        "kept": [{k: v for k, v in r.items()} for r in kept],
        "real": {"cells": table.height, "passA": nA, "passB": nB},
        "placebo": plac,
        "informative": informative,
        "portfolio_A": port_A,
        "portfolio_B": port_B,
    }
    (OUT / "dev_state.json").write_text(json.dumps(state, indent=2, default=str))

    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=250, float_precision=2):
        print("\nTop 40 cells by stage-A t:")
        print(table.sort("tA", descending=True).head(40).select(
            "symbol", "family", "param", "cond", "h", "side", "nA", "midA", "netA", "tA", "qA",
            "nB", "netB", "tB", "passB"))
        print("\nStage B survivors:")
        print(surv.sort("tA", descending=True).select(
            "symbol", "family", "param", "cond", "h", "side", "nA", "netA", "tA", "nB", "netB", "tB"))
    print(f"\nplacebo mean stage-B survivors {mean_plac_B:.1f} vs real {nB} -> "
          f"{'informative' if informative else 'UNINFORMATIVE'}")
    print(f"kept after de-duplication: {len(kept)}")
    print("portfolio half A (selection half, biased):", port_A)
    print("portfolio half B (confirmation half):", port_B)


def validate() -> None:
    """The registered validation shot: power from timestamps first, outcomes only if it passes."""
    from qlab.eventstudy import forward_returns

    state = json.loads((OUT / "dev_state.json").read_text())
    kept = state["kept"]
    result: dict = {"kept": len(kept)}
    if not state["informative"] or not kept:
        print("dev screen produced no informative survivors; validation not read")
        return
    feats, bases, _, _ = build("validation", state)
    lookup = {(b.symbol, b.family, b.param): i for i, b in enumerate(bases)}
    for row in kept:
        row["base"] = lookup[(row["symbol"], row["family"], row["param"])]

    # Power, from event timestamps only.
    days = set()
    for row in kept:
        idx, _, _ = cell_events(row, bases, feats)
        days.update(feats[row["symbol"]].tdrank[idx].tolist())
    port_B = state["portfolio_B"]
    expected = port_B["t_day"] * math.sqrt(len(days) / port_B["days"])
    result.update(val_event_days=len(days), B_days=port_B["days"], expected_t=expected)
    print(f"validation event days {len(days)}, expected t {expected:.2f} (rule >= {POWER_T})")
    if expected < POWER_T:
        result["verdict"] = "shot not taken (underpowered)"
        (OUT / "validation.json").write_text(json.dumps(result, indent=2))
        return

    # Real bid/ask fills + commission + slippage, validation-era cost model.
    tapes = []
    for s in SYMBOLS:
        rows = [r for r in kept if r["symbol"] == s]
        if not rows:
            continue
        bars = load_bars(s, "1m", split="validation").sort("ts")
        for row in rows:
            idx, direction, _ = cell_events(row, bases, feats)
            b = bases[row["base"]]
            cont = b.cont if row["side"] == 1 else not b.cont
            cost = CostModel.from_profiles(s, split="validation",
                                           slippage=SlippageModel(adverse_fraction=1.0 if cont else 0.5))
            ev = pl.DataFrame({"ts": pl.from_epoch(pl.Series(feats[s].ts[idx]), time_unit="us")
                               .dt.replace_time_zone("UTC"), "direction": direction.astype(np.int8)})
            fr = forward_returns(bars, ev, (row["h"],), cost=cost)
            tapes.append(fr.select("ts", "mid_bps", "net_bps").with_columns(
                cell=pl.lit(f"{s} {row['family']} {row['param']} {row['cond']} h{row['h']} s{row['side']}")))
    tape = pl.concat(tapes)
    tape.write_parquet(OUT / "validation_tape.parquet")
    daily = tape.group_by(pl.col("ts").dt.date()).agg(pl.col("net_bps").sum())["net_bps"].to_numpy()
    t = float(daily.mean() / daily.std(ddof=1) * math.sqrt(daily.size))
    result.update(trades=tape.height, days=int(daily.size), net_per_trade=float(tape["net_bps"].mean()),
                  mid_per_trade=float(tape["mid_bps"].mean()), bps_per_day=float(daily.mean()), t_day=t,
                  sharpe_ann=float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)))
    result["verdict"] = "PASS" if (t >= T_B and daily.mean() > 0) else "FAIL"
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dev", "validate"])
    parser.add_argument("--placebos", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "dev":
        dev(args.placebos)
    else:
        validate()


if __name__ == "__main__":
    main()
