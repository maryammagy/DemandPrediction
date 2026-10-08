import numpy as np
import pandas as pd

from demand_forecasting.config import MAX_LEAD, REVEAL_FRACTION_FLOOR
from demand_forecasting.data import LEAD_COLUMNS
from demand_forecasting.reveal_curve import RevealCurve

HIST_END = pd.Timestamp("2025-01-01")


def _visible(rows: list[tuple[int, str, float]], curve: np.ndarray) -> pd.DataFrame:
    """Visible-demand frame where every row reveals ``curve[L]`` of its final demand by lead L."""
    records = []
    for cluster, checkin, total in rows:
        records.append([cluster, pd.Timestamp(checkin), *(total * curve)])
    return pd.DataFrame(records, columns=["cluster_code", "checkin", *LEAD_COLUMNS])


def _curve(half_life: float) -> np.ndarray:
    """Fraction visible at lead L: 1 at L=0, decaying towards 0 for long leads."""
    return np.exp(-np.arange(MAX_LEAD + 1) / half_life)


def test_fit_recovers_a_known_per_cluster_curve():
    fast, slow = _curve(3.0), _curve(9.0)
    rows_fast = [(1, f"2024-06-{d:02d}", 100.0) for d in range(1, 15)]
    rows_slow = [(2, f"2024-06-{d:02d}", 50.0) for d in range(1, 15)]
    visible = pd.concat([_visible(rows_fast, fast), _visible(rows_slow, slow)], ignore_index=True)

    reveal = RevealCurve.fit(visible, HIST_END)

    np.testing.assert_allclose(reveal.per_cluster.loc[1].to_numpy(), fast)
    np.testing.assert_allclose(reveal.per_cluster.loc[2].to_numpy(), slow)
    np.testing.assert_allclose(reveal.global_curve.to_numpy(), (fast + slow) / 2)  # median of two


def test_stage_a_recovers_final_demand_when_the_curve_is_exact():
    curve = _curve(5.0)
    rows = [(7, f"2024-06-{d:02d}", 200.0 + d) for d in range(1, 20)]
    reveal = RevealCurve.fit(_visible(rows, curve), HIST_END)

    lead = 10
    partial = 200.0 * curve[lead]
    fraction = reveal.fraction(np.array([7]), np.array([lead]), np.array([0]))
    np.testing.assert_allclose(partial / fraction, 200.0, rtol=1e-9)


def test_unknown_clusters_fall_back_to_the_pooled_curve():
    curve = _curve(4.0)
    reveal = RevealCurve.fit(
        _visible([(1, "2024-06-01", 10.0), (1, "2024-06-02", 10.0)], curve), HIST_END
    )
    fraction = reveal.fraction(np.array([1, 999]), np.array([6, 6]))
    assert fraction[0] == fraction[1]  # cluster 999 was never seen -> pooled curve
    np.testing.assert_allclose(fraction[0], curve[6])


def test_fraction_is_floored_and_leads_are_clipped():
    reveal = RevealCurve.fit(_visible([(1, "2024-06-01", 10.0)], _curve(1.0)), HIST_END)
    far = reveal.fraction(np.array([1]), np.array([MAX_LEAD]))
    assert far[0] == REVEAL_FRACTION_FLOOR  # exp(-59) is far below the floor
    # leads beyond MAX_LEAD behave like MAX_LEAD instead of raising
    assert reveal.fraction(np.array([1]), np.array([500]))[0] == far[0]


def test_weekday_adjustment_scales_the_fraction():
    # weekend-style check-ins (Friday, pandas weekday 4) reveal less at a given lead
    normal, weekend = _curve(4.0), _curve(8.0)
    rows = []
    for week in range(10):
        monday = pd.Timestamp("2024-03-04") + pd.Timedelta(weeks=week)
        rows.append((1, monday, 100.0))
        rows.append((1, monday + pd.Timedelta(days=4), 100.0))
    frame = pd.concat(
        [
            _visible([(c, d, t) for c, d, t in rows if pd.Timestamp(d).dayofweek == 0], normal),
            _visible([(c, d, t) for c, d, t in rows if pd.Timestamp(d).dayofweek == 4], weekend),
        ],
        ignore_index=True,
    )
    reveal = RevealCurve.fit(frame, HIST_END)
    monday = reveal.fraction(np.array([1]), np.array([5]), np.array([0]))
    friday = reveal.fraction(np.array([1]), np.array([5]), np.array([4]))
    assert friday[0] > monday[0]  # `weekend` decays slower -> larger fraction visible at lead 5
    assert reveal.dow_adjustment.to_numpy().min() >= 0.3
    assert reveal.dow_adjustment.to_numpy().max() <= 3.0


def test_fit_ignores_check_ins_from_the_future():
    curve = _curve(4.0)
    past = _visible([(1, f"2024-06-{d:02d}", 100.0) for d in range(1, 10)], curve)
    future_a = _visible([(1, "2025-02-01", 100.0)], _curve(1.0))
    future_b = _visible([(1, "2025-02-01", 100.0)], _curve(50.0))
    a = RevealCurve.fit(pd.concat([past, future_a], ignore_index=True), HIST_END)
    b = RevealCurve.fit(pd.concat([past, future_b], ignore_index=True), HIST_END)
    pd.testing.assert_frame_equal(a.per_cluster, b.per_cluster)
    pd.testing.assert_series_equal(a.global_curve, b.global_curve)


def test_naive_fraction_uses_only_the_pooled_curve():
    curve = _curve(6.0)
    reveal = RevealCurve.fit(
        _visible([(1, "2024-06-01", 10.0), (2, "2024-06-02", 10.0)], curve), HIST_END
    )
    np.testing.assert_allclose(reveal.naive_fraction(np.array([0, 3, 10])), curve[[0, 3, 10]])
