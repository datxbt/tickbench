"""Build a point-in-time US small-cap universe and download its daily bars.

Poudel (2025) defines the universe as US-listed equities with market cap between
$300M and $2B and dollar volume above $500K/day, but names no index and ships no
constituent list. The S&P SmallCap 600 is the closest published proxy: it is the
$300M-$2B band by construction, it is investable, and - unlike the Russell 2000 -
Wikipedia carries its **full change history**, which is what makes a point-in-time
reconstruction possible at all.

The reconstruction walks the change table backwards from today's membership. For
every announced change ``(date, added, removed)`` the state *before* that date is
the state after it, minus the addition, plus the removal. That gives exact
membership back to the first row of the change table (2019-12-17); before that the
membership is frozen at its 2019-12 value, which is flagged in the report and only
affects the in-sample period.

Residual survivorship bias is *measured*, not assumed away: tickers that Yahoo no
longer serves are counted and reported, because those are exactly the names that
were acquired, delisted or went to zero.

Written to ``data/external/smallcap/``:
  ``universe_membership.parquet``   ticker x date membership matrix
  ``bars_daily.parquet``            OHLCV for every ticker ever in the index
  ``benchmarks.parquet``            SPY, IWM, ^VIX daily bars
  ``fetch_report.json``             what downloaded, what did not, and why
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "data" / "external" / "smallcap"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
UA = {"User-Agent": "Mozilla/5.0 (research; qlab)"}
BENCHMARKS = ["SPY", "IWM", "^VIX"]


def _wiki_tables() -> list[pd.DataFrame]:
    req = urllib.request.Request(WIKI, headers=UA)
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def _normalise(ticker: str) -> str:
    """Wikipedia writes class shares as ``BRK.B``; Yahoo wants ``BRK-B``."""
    return str(ticker).strip().upper().replace(".", "-")


def build_membership(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = _wiki_tables()
    current = tables[0]
    changes = tables[1]
    changes.columns = ["date", "add_t", "add_s", "rem_t", "rem_s", "reason"]
    changes["date"] = pd.to_datetime(changes["date"], format="mixed", errors="coerce")
    changes = changes.dropna(subset=["date"]).sort_values("date", ascending=False)

    today = {_normalise(t) for t in current["Symbol"]}

    # Walk backwards: state before a change = state after it - added + removed.
    # Keyed by the change date, holding membership as of the *day before* it.
    history: list[tuple[pd.Timestamp, set[str]]] = []
    members = set(today)
    for _, row in changes.iterrows():
        add, rem = row["add_t"], row["rem_t"]
        if isinstance(add, str) and add.strip():
            members.discard(_normalise(add))
        if isinstance(rem, str) and rem.strip():
            members.add(_normalise(rem))
        history.append((row["date"], set(members)))

    # Turn the event list into a daily membership matrix.
    days = pd.bdate_range(start, end)
    every_ticker = sorted(today.union(*[m for _, m in history]) if history else today)
    matrix = pd.DataFrame(False, index=days, columns=every_ticker)

    # history is newest-first; history[i] holds membership before history[i][0].
    for day in days:
        state = today
        for change_date, before in history:
            if day < change_date:
                state = before
            else:
                break
        matrix.loc[day, sorted(state)] = True

    return matrix, changes


def download(tickers: list[str], start: date, end: date, chunk: int = 40) -> tuple[pd.DataFrame, list[str]]:
    import yfinance as yf

    frames, failed = [], []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="column",
            )
        except Exception as exc:  # pragma: no cover - network
            print(f"  batch {i//chunk}: {exc}", file=sys.stderr)
            failed.extend(batch)
            continue
        if raw is None or raw.empty:
            failed.extend(batch)
            continue
        got = _tidy(raw, batch)
        if got.empty:
            failed.extend(batch)
        else:
            frames.append(got)
            missing = sorted(set(batch) - set(got["ticker"].unique()))
            failed.extend(missing)
        print(f"  {i + len(batch)}/{len(tickers)} tickers", flush=True)
        time.sleep(0.6)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, sorted(set(failed))


def _tidy(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    """Yahoo's wide multi-index frame -> long ``date, ticker, o/h/l/c/adj/volume``."""
    fields = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, batch[:1]])
    pieces = []
    for src, dst in fields.items():
        if src not in raw.columns.get_level_values(0):
            continue
        block = raw[src].stack(future_stack=True).rename(dst)
        pieces.append(block)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).reset_index()
    out.columns = ["date", "ticker", *[c for c in out.columns[2:]]]
    out = out.dropna(subset=["close"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--limit", type=int, default=0, help="cap ticker count (smoke test)")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    OUT.mkdir(parents=True, exist_ok=True)

    print("building point-in-time membership from the S&P 600 change history")
    matrix, changes = build_membership(start, end)
    tickers = list(matrix.columns)
    if args.limit:
        tickers = tickers[: args.limit]
        matrix = matrix[tickers]
    print(f"  {len(tickers)} tickers ever in the index over the window")
    print(f"  membership exact back to {changes['date'].min().date()}")

    matrix.to_parquet(OUT / "universe_membership.parquet")
    changes.to_parquet(OUT / "index_changes.parquet")

    print(f"downloading daily bars for {len(tickers)} tickers")
    bars, failed = download(tickers, start, end)
    if not bars.empty:
        bars.to_parquet(OUT / "bars_daily.parquet", index=False)

    print("downloading benchmarks")
    bench, bench_failed = download(BENCHMARKS, start, end, chunk=3)
    if not bench.empty:
        bench.to_parquet(OUT / "benchmarks.parquet", index=False)

    report = {
        "start": args.start,
        "end": args.end,
        "tickers_requested": len(tickers),
        "tickers_downloaded": int(bars["ticker"].nunique()) if not bars.empty else 0,
        "tickers_failed": failed,
        "n_failed": len(failed),
        "membership_exact_from": str(changes["date"].min().date()),
        "rows": int(len(bars)),
        "benchmarks_failed": bench_failed,
    }
    (OUT / "fetch_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "tickers_failed"}, indent=2))


if __name__ == "__main__":
    main()
