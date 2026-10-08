import pandas as pd

from demand_forecasting.config import MAX_LEAD
from demand_forecasting.synthetic import SyntheticConfig, generate, write_dataset

CUTOFF = pd.Timestamp("2025-11-21")


def test_schema_and_ranges(synthetic):
    logs, evaluation = synthetic["search_data"], synthetic["evaluation"]
    for frame in (logs, evaluation):
        assert list(frame.columns) == ["log_date", "city_code", "checkin", "search_count"]
        assert (frame["search_count"] > 0).all()
        lead = (frame["checkin"] - frame["log_date"]).dt.days
        assert lead.between(0, MAX_LEAD).all()
    assert set(synthetic["cities"].columns) == {"city_code", "province_code", "lat", "long"}


def test_history_is_complete_and_the_target_month_is_only_partially_observed(synthetic):
    logs, evaluation, truth = (
        synthetic["search_data"],
        synthetic["evaluation"],
        synthetic["target_truth"],
    )
    assert logs["checkin"].max() <= CUTOFF
    assert evaluation["checkin"].min() > CUTOFF
    assert evaluation["log_date"].max() <= CUTOFF  # nothing logged after the forecast origin
    assert truth["checkin"].nunique() == 30

    seen = evaluation.groupby(["city_code", "checkin"])["search_count"].sum()
    final = truth.set_index(["city_code", "checkin"])["demand"]
    assert (seen <= final.reindex(seen.index)).all()  # partial <= eventual
    assert seen.sum() < final.sum()  # genuinely censored


def test_generation_is_deterministic_and_seed_dependent(holidays):
    config = SyntheticConfig(n_cities=5, n_provinces=2, seed=11, start="2025-06-01", scale=0.3)
    a, b = generate(config, holidays), generate(config, holidays)
    pd.testing.assert_frame_equal(a["search_data"], b["search_data"])
    other = generate(
        SyntheticConfig(n_cities=5, n_provinces=2, seed=12, start="2025-06-01", scale=0.3), holidays
    )
    assert not a["search_data"].equals(other["search_data"])


def test_write_dataset_creates_the_expected_files(tmp_path):
    config = SyntheticConfig(n_cities=4, n_provinces=2, start="2025-08-01", scale=0.3)
    paths = write_dataset(tmp_path / "synthetic", config)
    assert {p.name for p in paths.values()} == {
        "search_data.csv",
        "evaluation.csv",
        "cities.csv",
        "target_truth.csv",
    }
    assert all(p.stat().st_size > 0 for p in paths.values())
