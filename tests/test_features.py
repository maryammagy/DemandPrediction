import numpy as np
import pandas as pd
import pytest

from demand_forecasting.config import MAX_LEAD
from demand_forecasting.data import FINAL_DEMAND_COLUMN, LEAD_COLUMNS
from demand_forecasting.features import (
    FEATURES,
    FittedComponents,
    HistoryStats,
    attach_prior_year,
    build_inference_frame,
    build_training_panel,
)
from demand_forecasting.jalali import JalaliMonth

WINDOW = JalaliMonth(1404, 8)  # Aban 1404: 2025-10-23 .. 2025-11-21, cutoff 2025-10-22
LEADS = (1, 5, 14, 30)


def test_history_stats_per_cluster_and_weekday():
    # cluster 1: Mondays have demand 10 and 30, Tuesdays 100
    visible = pd.DataFrame(
        {
            "cluster_code": [1, 1, 1],
            "checkin": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-07"]),  # Mon, Mon, Tue
            FINAL_DEMAND_COLUMN: [10.0, 30.0, 100.0],
        }
    )
    stats = HistoryStats.fit(visible, pd.Timestamp("2025-02-01"))
    assert stats.overall.loc[0, "cluster_hist_mean"] == pytest.approx(140 / 3)
    monday = stats.by_weekday.query("dow_raw == 0").iloc[0]
    assert monday["cluster_dow_hist_mean"] == 20.0
    assert np.isclose(monday["cluster_dow_hist_std"], np.std([10, 30], ddof=1))


def test_history_attach_falls_back_for_unseen_cluster_and_weekday():
    visible = pd.DataFrame(
        {
            "cluster_code": [1, 1],
            "checkin": pd.to_datetime(["2025-01-06", "2025-01-13"]),
            FINAL_DEMAND_COLUMN: [10.0, 30.0],
        }
    )
    stats = HistoryStats.fit(visible, pd.Timestamp("2025-02-01"))
    rows = pd.DataFrame(
        {
            "cluster_code": [1, 1, 2],
            "checkin": pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-03"]),  # Mon, Tue, Mon
        }
    )
    out = stats.attach(rows)
    assert out["cluster_dow_hist_mean"].tolist() == [
        20.0,
        20.0,
        0.0,
    ]  # Tue -> cluster mean; cluster 2 -> 0
    assert out["cluster_hist_mean_log"].iloc[2] == 0.0


def test_prior_year_picks_the_nearest_date_within_tolerance_and_never_the_future():
    visible = pd.DataFrame(
        {
            "cluster_code": [1, 1, 1],
            "checkin": pd.to_datetime(["2024-03-10", "2024-03-12", "2025-03-09"]),
            FINAL_DEMAND_COLUMN: [100.0, 200.0, 999.0],
        }
    )
    rows = pd.DataFrame(
        {"cluster_code": [1, 1], "checkin": pd.to_datetime(["2025-03-11", "2025-03-20"])}
    )
    # hist_end excludes the 2025 row, so the 999 can never be returned
    out = attach_prior_year(rows, visible, hist_end=pd.Timestamp("2025-03-01"))
    # 2025-03-11 - 365d = 2024-03-11 (leap year offset) -> nearest of 03-10 / 03-12 within 6 days
    assert out.sort_values("checkin")["prior_year_demand"].iloc[0] in (100.0, 200.0)
    # 2025-03-20 - 365d = 2024-03-20 is 8 days from the nearest history -> beyond tolerance -> 0
    assert out.sort_values("checkin")["prior_year_demand"].iloc[1] == 0.0


def test_training_panel_has_every_feature_and_replays_each_lead(visible, holidays):
    components = FittedComponents.fit(visible, WINDOW.start)
    panel = build_training_panel(visible, components, holidays, LEADS)

    assert set(FEATURES) <= set(panel.columns)
    assert not panel[FEATURES].isna().any().any()
    assert sorted(panel["lead"].unique()) == sorted(LEADS)
    n_history = (visible["checkin"] < WINDOW.start).sum()
    assert len(panel) == n_history * len(LEADS)
    assert panel["checkin"].max() < WINDOW.start
    # replayed partial demand never exceeds the final demand it will be compared with
    assert (panel["partial_observed"] <= panel["target"] + 1e-9).all()


def test_inference_frame_observes_only_what_was_logged_before_the_cutoff(visible, holidays):
    components = FittedComponents.fit(visible, WINDOW.start)
    pairs = (
        visible[["cluster_code"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"checkin": pd.date_range(WINDOW.start, WINDOW.end)}), how="cross")
    )
    frame = build_inference_frame(pairs, WINDOW.cutoff, visible, components, holidays)

    assert len(frame) == len(pairs)
    assert frame["lead"].min() == 1 and frame["lead"].max() == (WINDOW.end - WINDOW.cutoff).days
    lookup = visible.set_index(["cluster_code", "checkin"])
    sample = frame.sample(20, random_state=0)
    for _, row in sample.iterrows():
        expected = lookup.loc[(row["cluster_code"], row["checkin"]), LEAD_COLUMNS[int(row["lead"])]]
        assert row["partial_observed"] == expected
    assert (frame["partial_observed"] <= frame[FINAL_DEMAND_COLUMN]).all()
    assert not frame[FEATURES].isna().any().any()


def test_features_do_not_leak_logs_written_after_the_cutoff(visible, holidays):
    """The strongest leakage guard: searches logged *after* the cutoff must not move any feature.

    We inflate, for every check-in in the scored window, all demand that was logged after the
    cutoff (lead < checkin - cutoff). Features, Stage A and the training panel must be identical;
    only the (hidden) truth may change.
    """
    leaky = visible.copy()
    window_rows = (leaky["checkin"] >= WINDOW.start) & (leaky["checkin"] <= WINDOW.end)
    lead_of_row = (leaky["checkin"] - WINDOW.cutoff).dt.days.to_numpy()
    for lead in range(MAX_LEAD):
        affected = window_rows.to_numpy() & (lead_of_row > lead)
        leaky.loc[affected, LEAD_COLUMNS[lead]] += 1_000.0  # extra searches logged after cutoff

    def run(frame):
        components = FittedComponents.fit(frame, WINDOW.start)
        pairs = (
            frame[["cluster_code"]]
            .drop_duplicates()
            .merge(pd.DataFrame({"checkin": pd.date_range(WINDOW.start, WINDOW.end)}), how="cross")
        )
        cells = build_inference_frame(pairs, WINDOW.cutoff, frame, components, holidays)
        panel = build_training_panel(frame, components, holidays, LEADS)
        return cells, panel

    clean_cells, clean_panel = run(visible)
    leaky_cells, leaky_panel = run(leaky)

    assert not clean_cells[FINAL_DEMAND_COLUMN].equals(
        leaky_cells[FINAL_DEMAND_COLUMN]
    )  # truth moved
    pd.testing.assert_frame_equal(clean_cells[FEATURES], leaky_cells[FEATURES])
    pd.testing.assert_frame_equal(
        clean_cells[["stageA_estimate"]], leaky_cells[["stageA_estimate"]]
    )
    pd.testing.assert_frame_equal(clean_panel, leaky_panel)
