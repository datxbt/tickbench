"""The cost model.

The tests that matter here are the ones that pin an *assumption* rather than a
calculation: that turning slippage off reproduces the spread-plus-commission
number exactly, so the difference between the two is always visible; and that
nothing quietly changes the adverse fraction, which is the number carrying the
most weight in the whole model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from qlab.costs import CHASING, NO_SLIPPAGE, CostModel, SlippageModel

UTC = timezone.utc


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Synthetic profiles: flat 1.0 pip spread, flat 2.0 pip drift at 250 ms.

    Flat on purpose - a constant makes every arithmetic assertion below readable
    by hand, and the hour-keyed behaviour is tested separately with a profile
    that varies.
    """
    spread = pl.DataFrame(
        {
            "symbol": ["EURUSD"] * 24 + ["USDJPY"] * 24 + ["XAUUSD"] * 24,
            "year": [2024] * 72,
            "month": [6] * 72,
            "hour": list(range(24)) * 3,
            "is_sunday": [False] * 72,
            "n_bars": [100] * 72,
            "n_ticks": [1000] * 72,
            "spread_mean_pips": [1.0] * 72,
            "spread_p50_pips": [1.0] * 72,
            "spread_p95_pips": [2.0] * 72,
            "spread_max_pips": [5.0] * 72,
        }
    )
    latency = pl.DataFrame(
        {
            "symbol": ["EURUSD"] * 24 + ["USDJPY"] * 24 + ["XAUUSD"] * 24,
            "year": [2024] * 72,
            "month": [6] * 72,
            "hour": list(range(24)) * 3,
            "horizon_ms": [250] * 72,
            "n_anchors": [1000] * 72,
            "drift_mean_pips": [2.0] * 72,
            "drift_p50_pips": [1.0] * 72,
            "drift_p95_pips": [6.0] * 72,
            "zero_share": [0.5] * 72,
        }
    )
    # Match the schema the real builder writes, so the fixture cannot pass a
    # test that the built profile would fail.
    cast = {"year": pl.Int32, "month": pl.Int8}
    spread = spread.cast(cast)
    latency = latency.cast(cast | {"horizon_ms": pl.Int32})

    spread_path = tmp_path / "spread.parquet"
    latency_path = tmp_path / "latency.parquet"
    spread.write_parquet(spread_path)
    latency.write_parquet(latency_path)
    monkeypatch.setattr("qlab.paths.SPREAD_PROFILE_PATH", spread_path)
    monkeypatch.setattr("qlab.paths.LATENCY_PROFILE_PATH", latency_path)
    return spread_path, latency_path


def _bars(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    """Minimal bar frame: ts_open, close and a realized spread in price units."""
    return pl.DataFrame(
        {
            "ts_open": [datetime.fromisoformat(r[0]).replace(tzinfo=UTC) for r in rows],
            "ts": [datetime.fromisoformat(r[0]).replace(tzinfo=UTC) for r in rows],
            "close": [r[1] for r in rows],
            "spread_mean": [r[2] for r in rows],
        },
        schema={
            "ts_open": pl.Datetime("us", "UTC"),
            "ts": pl.Datetime("us", "UTC"),
            "close": pl.Float64,
            "spread_mean": pl.Float64,
        },
    )


def test_turning_slippage_off_reproduces_spread_plus_commission(profiles):
    """The anchor. Stage 0 costed a round turn as spread + commission; that
    number has to survive exactly, or the two stages cannot be compared."""
    model = CostModel.from_profiles("EURUSD", slippage=NO_SLIPPAGE)

    assert model.spread_pips() == pytest.approx(1.0)
    assert model.commission_pips() == pytest.approx(0.5)
    assert model.round_turn_pips() == pytest.approx(1.5)


def test_slippage_is_a_stated_fraction_of_measured_drift(profiles):
    """Drift is measured; what share of it is adverse is an assumption, and the
    assumption has to move the answer visibly and linearly."""
    default = CostModel.from_profiles("EURUSD")
    chasing = CostModel.from_profiles("EURUSD", slippage=CHASING)

    assert default.drift_pips() == pytest.approx(2.0)
    # Half of a 2.0 pip drift, on each of two fills.
    assert default.round_turn_pips() == pytest.approx(1.0 + 0.5 + 2 * 1.0)
    assert chasing.round_turn_pips() == pytest.approx(1.0 + 0.5 + 2 * 2.0)

    off = CostModel.from_profiles("EURUSD", slippage=NO_SLIPPAGE)
    assert chasing.round_turn_pips() - off.round_turn_pips() == pytest.approx(4.0)


def test_a_round_turn_crosses_the_spread_once_and_slips_twice(profiles):
    model = CostModel.from_profiles("EURUSD")
    parts = model.breakdown(price=1.08)

    assert parts["spread_pips"] == pytest.approx(1.0)  # in at ask, out at bid
    assert parts["commission_pips"] == pytest.approx(0.5)  # both sides
    assert parts["slippage_pips"] == pytest.approx(2.0)  # both fills
    assert parts["round_turn_pips"] == pytest.approx(3.5)
    assert sum(
        parts[k] for k in ("spread_share", "commission_share", "slippage_share")
    ) == pytest.approx(1.0)


def test_bps_is_the_cross_instrument_unit(profiles):
    """Pips are not comparable between gold and EURUSD; bps of price are."""
    model = CostModel.from_profiles("EURUSD")
    expected = model.round_turn_pips() * model.spec.pip / 1.08 * 10_000

    assert model.round_turn_bps(price=1.08) == pytest.approx(expected)


def test_a_jpy_symbol_needs_the_rate_to_price_commission(profiles):
    """A JPY pip is worth a rate-dependent number of dollars, so the commission
    in pips is not a constant - and asking for it without a price is an error
    rather than a silent 1.0."""
    model = CostModel.from_profiles("USDJPY")

    with pytest.raises(ValueError, match="depends on the rate"):
        model.commission_pips()

    assert model.commission_pips(155.0) == pytest.approx(0.775, rel=1e-3)
    assert model.commission_pips(100.0) < model.commission_pips(155.0)


def test_realized_spread_comes_from_the_bar_not_the_profile(profiles):
    """The default source is the quote that was actually there, which is both
    more accurate than a profile and free of lookahead."""
    model = CostModel.from_profiles("EURUSD")
    bars = _bars(
        [
            ("2024-06-03T10:00:00", 1.08, 0.00005),  # half a pip, realized
            ("2024-06-03T11:00:00", 1.08, 0.00030),  # three pips
        ]
    )

    realized = model.with_costs(bars)
    profiled = model.with_costs(bars, spread_source="profile")

    assert realized["spread_pips"].to_list() == pytest.approx([0.5, 3.0])
    assert profiled["spread_pips"].to_list() == pytest.approx([1.0, 1.0])


def test_side_cost_is_half_the_round_turn(profiles):
    """The composable primitive: a vectorized backtest multiplies it by |dposition|."""
    model = CostModel.from_profiles("EURUSD")
    costed = model.with_costs(_bars([("2024-06-03T10:00:00", 1.08, 0.0001)]))

    assert costed["side_cost_pips"][0] * 2 == pytest.approx(costed["round_turn_pips"][0])
    assert costed["round_turn_pips"][0] == pytest.approx(1.0 + 0.5 + 2 * 1.0)


def test_costs_are_priced_by_the_interval_open_not_the_bar_label(profiles, tmp_path, monkeypatch):
    """The same trap right-edge labelling sets for session flags.

    A bar covering [20:59, 21:00) is labelled 21:00. Pricing it from the label
    would charge it the rollover hour's spread, which none of its ticks paid.
    """
    spread = pl.read_parquet(tmp_path / "spread.parquet").with_columns(
        spread_mean_pips=pl.when(pl.col("hour") == 21).then(50.0).otherwise(1.0)
    )
    spread.write_parquet(tmp_path / "spread.parquet")

    model = CostModel.from_profiles("EURUSD")
    bars = _bars([("2024-06-03T20:59:00", 1.08, 0.0001)])
    costed = model.with_costs(bars, spread_source="profile")

    assert costed["spread_pips"][0] == pytest.approx(1.0)


def test_stress_widens_the_spread_without_touching_the_rest(profiles):
    model = CostModel.from_profiles("EURUSD")
    stressed = model.stressed(spread_multiplier=3.0, extra_slippage_pips=0.25)

    assert stressed.spread_pips() == pytest.approx(3.0)
    assert stressed.commission_pips() == pytest.approx(model.commission_pips())
    # 3 pips spread + 0.5 commission + 2 x (1.0 + 0.25)
    assert stressed.round_turn_pips() == pytest.approx(6.0)
    assert model.round_turn_pips() == pytest.approx(3.5)  # original is unchanged


def test_changing_latency_requires_rebuilding_from_the_profile(profiles):
    """A different horizon is a different measurement, not a scaling of this one."""
    model = CostModel.from_profiles("EURUSD")
    with pytest.raises(ValueError, match="rebuilding from the profile"):
        model.stressed(latency_ms=1000)


def test_an_unmeasured_latency_is_refused(profiles):
    with pytest.raises(ValueError, match="measured horizon"):
        SlippageModel(latency_ms=300)


def test_the_adverse_fraction_is_a_share(profiles):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SlippageModel(adverse_fraction=1.5)


def test_turnover_is_what_makes_a_small_cost_large(profiles):
    """The number to check a strategy's gross return against."""
    model = CostModel.from_profiles("EURUSD")
    once = model.annual_drag_bps(price=1.08, round_turns_per_day=1)
    ten = model.annual_drag_bps(price=1.08, round_turns_per_day=10)

    assert ten == pytest.approx(10 * once)
    assert once == pytest.approx(model.round_turn_bps(price=1.08) * 252)


def test_the_window_selects_which_period_is_being_costed(profiles, tmp_path):
    """Spread has compressed by an order of magnitude since 2020, so a model
    fitted over everything is wrong at both ends."""
    spread = pl.read_parquet(tmp_path / "spread.parquet")
    older = spread.with_columns(year=pl.lit(2020, pl.Int32), spread_mean_pips=pl.lit(9.0))
    pl.concat([older, spread]).write_parquet(tmp_path / "spread.parquet")
    latency = pl.read_parquet(tmp_path / "latency.parquet")
    pl.concat([latency.with_columns(year=pl.lit(2020, pl.Int32)), latency]).write_parquet(
        tmp_path / "latency.parquet"
    )

    from datetime import date

    wide = CostModel.from_profiles("EURUSD", until=date(2020, 12, 31))
    tight = CostModel.from_profiles("EURUSD", trailing_months=1)

    assert wide.spread_pips() == pytest.approx(9.0)
    assert tight.spread_pips() == pytest.approx(1.0)


def test_describe_states_every_assumption(profiles):
    """A report or a log line has to be able to show what it assumed."""
    text = CostModel.from_profiles("XAUUSD", slippage=CHASING).describe()

    assert "XAUUSD" in text
    assert "1 x drift over 250 ms" in text
    assert "$3.5/lot/side" in text


def test_a_model_from_the_wrong_period_is_refused(profiles):
    """Costing 2020 trades at 2026 spreads is close to a factor of two on gold
    and USTEC, and nothing about the output would look wrong."""
    model = CostModel.from_profiles("EURUSD")  # profile covers 2024-06 only
    stale = _bars([("2020-06-03T10:00:00", 1.08, 0.0001)])

    with pytest.raises(ValueError, match="was measured over"):
        model.with_costs(stale)


def test_bars_straddling_the_window_edge_warn_rather_than_fail(profiles):
    """Partial overlap is usually a warmup period, which is legitimate - but the
    costs outside the window are extrapolated and the caller should know."""
    model = CostModel.from_profiles("EURUSD")
    straddling = _bars(
        [
            ("2024-05-28T10:00:00", 1.08, 0.0001),  # before the profile starts
            ("2024-06-03T10:00:00", 1.08, 0.0001),
        ]
    )

    with pytest.warns(UserWarning, match="outside the cost"):
        costed = model.with_costs(straddling)

    assert costed.height == 2
