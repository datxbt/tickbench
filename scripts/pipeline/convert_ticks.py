"""Convert raw Exness tick CSVs to partitioned Parquet.

Usage
-----
    python scripts/pipeline/convert_ticks.py                  # all symbols, skip up-to-date files
    python scripts/pipeline/convert_ticks.py -s XAUUSD USTEC  # subset
    python scripts/pipeline/convert_ticks.py --force          # rebuild everything
    python scripts/pipeline/convert_ticks.py -w 8             # worker processes
"""

from __future__ import annotations

import argparse
import sys
import time

from qlab import convert
from qlab.symbols import ALL_SYMBOLS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--symbols", nargs="+", default=list(ALL_SYMBOLS))
    parser.add_argument("-w", "--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="reconvert up-to-date files")
    args = parser.parse_args()

    # Self-heal first: a previous run that died before writing the manifest
    # leaves converted files permanently unlisted, since they are no longer stale.
    _, recovered = convert.reconcile_manifest()
    if recovered:
        print(f"recovered {recovered} manifest rows from parquet already on disk", flush=True)

    sources = convert.discover(args.symbols)
    todo = sources if args.force else [s for s in sources if convert.is_stale(s)]
    print(
        f"{len(sources)} source files found, {len(todo)} to convert "
        f"({len(sources) - len(todo)} already up to date)",
        flush=True,
    )
    if not todo:
        return 0

    started = time.perf_counter()
    records: list[dict] = []
    failures: list[dict] = []

    for done, record in enumerate(
        convert.run(args.symbols, force=args.force, workers=args.workers), start=1
    ):
        records.append(record)
        label = f"{record['symbol']} {record['year']}-{record['month']:02d}"
        if record.get("status") == "ok":
            ratio = record["src_bytes"] / max(record["out_bytes"], 1)
            print(
                f"[{done:>3}/{len(todo)}] {label}  {record['rows']:>10,} rows  "
                f"{record['out_bytes'] / 1e6:>7.1f} MB  {ratio:>4.1f}x  "
                f"{record['elapsed_s']:>6.1f}s",
                flush=True,
            )
        else:
            failures.append(record)
            print(f"[{done:>3}/{len(todo)}] {label}  FAILED  {record['error']}", flush=True)

    manifest = convert.write_manifest(records)
    ok = [r for r in records if r.get("status") == "ok"]
    src_gb = sum(r["src_bytes"] for r in ok) / 1e9
    out_gb = sum(r["out_bytes"] for r in ok) / 1e9

    print(
        f"\nconverted {len(ok)}/{len(todo)} files, {sum(r['rows'] for r in ok):,} rows, "
        f"{src_gb:.1f} GB CSV -> {out_gb:.1f} GB parquet "
        f"({src_gb / max(out_gb, 1e-9):.1f}x) in {(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    print(f"manifest: {manifest}", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILURES:", flush=True)
        for record in failures:
            print(f"  {record['source_file']}: {record['error']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
