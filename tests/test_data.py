import numpy as np
import pandas as pd
import pytest

from demand_forecasting.config import MAX_LEAD
from demand_forecasting.data import (
    LEAD_COLUMNS,
    aggregate_to_clusters,
    build_visible_demand,
    read_search_logs,
)


def _logs(rows):
    return pd.DataFrame(
        rows, columns=["log_date", "cluster_code", "checkin", "search_count"]
    ).astype({"log_date": "datetime64[ns]", "checkin": "datetime64[ns]"})


def test_visible_demand_is_the_reverse_cumulative_sum_over_leads():
    # check-in 2025-01-10: 3 searches 5 days ahead, 4 searches 2 days ahead, 1 on the day
    logs = _logs(
        [
            ("2025-01-05", 1, "2025-01-10", 3),
            ("2025-01-08", 1, "2025-01-10", 4),
            ("2025-01-10", 1, "2025-01-10", 1),
        ]
    )
    clusters = pd.DataFrame({"city_code": [1], "cluster_code": [1]})
    visible = build_visible_demand(logs, clusters).iloc[0]

    assert visible["visible_at_lead_0"] == 8  # final demand
    assert visible["visible_at_lead_1"] == 7  # everything logged 1+ days ahead
    assert visible["visible_at_lead_2"] == 7
    assert visible["visible_at_lead_3"] == 3  # only the lead-5 searches are >= 3 days out
    assert visible["visible_at_lead_5"] == 3
    assert visible["visible_at_lead_6"] == 0


def test_grid_is_zero_filled_for_every_cluster_and_date():
    logs = _logs([("2025-01-09", 1, "2025-01-10", 2), ("2025-01-11", 1, "2025-01-12", 5)])
    clusters = pd.DataFrame({"city_code": [1, 2], "cluster_code": [1, 2]})  # cluster 2 is silent
    visible = build_visible_demand(logs, clusters)

    assert len(visible) == 2 * 2  # 2 clusters x 2 distinct check-in dates
    silent = visible[visible["cluster_code"] == 2]
    assert (silent[LEAD_COLUMNS] == 0).all().all()
    assert not visible[LEAD_COLUMNS].isna().any().any()


def test_logs_outside_the_lead_range_are_ignored():
    logs = _logs(
        [
            ("2025-01-11", 1, "2025-01-10", 7),  # lead -1: after the stay
            ("2024-10-01", 1, "2025-01-10", 9),  # lead > 59
            ("2025-01-09", 1, "2025-01-10", 2),
        ]
    )
    clusters = pd.DataFrame({"city_code": [1], "cluster_code": [1]})
    visible = build_visible_demand(logs, clusters)
    assert visible.loc[0, "visible_at_lead_0"] == 2
    assert len(LEAD_COLUMNS) == MAX_LEAD + 1


def test_cities_are_summed_into_clusters():
    search = pd.DataFrame(
        {
            "log_date": pd.to_datetime(["2025-01-01"] * 3),
            "city_code": [10, 11, 12],
            "checkin": pd.to_datetime(["2025-01-05"] * 3),
            "search_count": [1, 2, 4],
        }
    )
    mapping = pd.DataFrame({"city_code": [10, 11, 12], "cluster_code": [10, 10, 12]})
    out = aggregate_to_clusters(search, mapping).set_index("cluster_code")["search_count"]
    assert out.to_dict() == {10: 3, 12: 4}


def test_unmapped_cities_raise_a_clear_error():
    search = pd.DataFrame(
        {
            "log_date": pd.to_datetime(["2025-01-01"]),
            "city_code": [99],
            "checkin": pd.to_datetime(["2025-01-05"]),
            "search_count": [1],
        }
    )
    with pytest.raises(ValueError, match="no cluster"):
        aggregate_to_clusters(search, pd.DataFrame({"city_code": [1], "cluster_code": [1]}))


def test_read_search_logs_validates_input(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("log_date,city_code,checkin,search_count\n2025-01-01,1,2025-01-05,3\n")
    assert len(read_search_logs(good)) == 1

    negative = tmp_path / "negative.csv"
    negative.write_text("log_date,city_code,checkin,search_count\n2025-01-01,1,2025-01-05,-3\n")
    with pytest.raises(ValueError, match="negative"):
        read_search_logs(negative)

    missing = tmp_path / "missing.csv"
    missing.write_text("log_date,city_code,checkin\n2025-01-01,1,2025-01-05\n")
    with pytest.raises(ValueError):
        read_search_logs(missing)


def test_visible_demand_is_monotone_in_lead(visible):
    values = visible[LEAD_COLUMNS].to_numpy()
    assert (np.diff(values, axis=1) <= 0).all()  # earlier lead = more demand visible
