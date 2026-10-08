import numpy as np
import pandas as pd

from demand_forecasting.clustering import (
    ClusteringConfig,
    cluster_cities,
    haversine_km,
    monthly_demand_vectors,
)

# Jalali months 1-3 of 1404 are 2025-03-21 .. 2025-06-21; use one check-in per month.
MONTH_DATES = pd.to_datetime(["2025-03-25", "2025-04-25", "2025-05-25"])


def _world(rows):
    """rows: (city_code, province_code, lat, long, [searches in month 1, 2, 3])."""
    cities = pd.DataFrame(
        [r[:4] for r in rows], columns=["city_code", "province_code", "lat", "long"]
    )
    logs = []
    for code, _, _, _, monthly in rows:
        for date, count in zip(MONTH_DATES, monthly, strict=True):
            if count:
                logs.append((date - pd.Timedelta(days=3), code, date, count))
    search = pd.DataFrame(logs, columns=["log_date", "city_code", "checkin", "search_count"])
    return cities, search


def _cluster_of(result: pd.DataFrame) -> dict[int, int]:
    return dict(zip(result["city_code"], result["cluster_code"], strict=True))


def test_haversine_known_distance():
    # Tehran -> Mashhad is roughly 740 km as the crow flies
    assert 700 < float(haversine_km(35.69, 51.39, 36.30, 59.60)) < 780
    assert float(haversine_km(10.0, 20.0, 10.0, 20.0)) == 0.0


def test_monthly_vectors_pick_the_reference_year():
    cities, search = _world([(1, 10, 35.0, 51.0, [5, 6, 7])])
    matrix, months, year = monthly_demand_vectors(cities, search)
    assert year == 1404 and months == [1, 2, 3]
    assert matrix.tolist() == [[5, 6, 7]]


def test_strong_cities_stay_alone_and_a_sparse_one_joins_its_nearest_neighbour():
    cities, search = _world(
        [
            (1, 10, 35.00, 51.00, [900, 900, 900]),  # strong
            (2, 10, 35.10, 51.00, [900, 900, 900]),  # strong, 11 km from city 3
            (3, 10, 35.20, 51.00, [10, 10, 10]),  # sparse: nearest is city 2, not city 1
            (4, 10, 40.00, 51.00, [900, 900, 900]),  # strong, far away
        ]
    )
    result = cluster_cities(cities, search, ClusteringConfig(floor=500))
    mapping = _cluster_of(result)
    assert mapping[3] == mapping[2]
    assert len({mapping[1], mapping[2], mapping[4]}) == 3


def test_a_single_empty_month_triggers_a_merge():
    # city 2 looks fine on average (1,000/month) but has a month below the floor
    cities, search = _world(
        [
            (1, 10, 35.0, 51.0, [900, 900, 900]),
            (2, 10, 35.1, 51.0, [2000, 2000, 5]),
        ]
    )
    mapping = _cluster_of(cluster_cities(cities, search, ClusteringConfig(floor=500)))
    assert mapping[1] == mapping[2]


def test_clusters_never_cross_provinces_and_every_cluster_clears_the_floor():
    rng = np.random.default_rng(0)
    rows = [
        (
            100 + i,
            10 + i % 3,
            30 + rng.random() * 5,
            50 + rng.random() * 5,
            list(rng.integers(0, 400, 3)),
        )
        for i in range(30)
    ]
    cities, search = _world(rows)
    result = cluster_cities(cities, search, ClusteringConfig(floor=500))
    merged = result.merge(cities[["city_code", "province_code"]], on="city_code")
    assert (merged.groupby("cluster_code")["province_code"].nunique() == 1).all()

    vectors, _, _ = monthly_demand_vectors(cities, search)
    per_cluster = (
        pd.DataFrame(vectors, index=result["cluster_code"].to_numpy()).groupby(level=0).sum()
    )
    totals_by_province = (
        pd.DataFrame(vectors, index=cities["province_code"].to_numpy()).groupby(level=0).sum()
    )
    for cluster, group in merged.groupby("cluster_code"):
        province = group["province_code"].iloc[0]
        feasible = totals_by_province.loc[province].min() >= 500
        if feasible:  # a province that cannot reach the floor even fully merged is exempt
            assert per_cluster.loc[cluster].min() >= 500


def test_clustering_is_deterministic():
    rng = np.random.default_rng(1)
    rows = [
        (i, 7, 30 + rng.random() * 3, 50 + rng.random() * 3, list(rng.integers(0, 300, 3)))
        for i in range(20)
    ]
    cities, search = _world(rows)
    a = cluster_cities(cities, search, ClusteringConfig(floor=500))
    b = cluster_cities(cities, search, ClusteringConfig(floor=500))
    pd.testing.assert_frame_equal(a, b)


def test_zero_volume_centroid_quirk_and_its_fix():
    """Two zero-demand cities merge; the legacy (0, 0) centroid then drags them to the city
    nearest the Gulf of Guinea instead of to their real neighbour."""
    cities, search = _world(
        [
            (1, 10, 10.0, 10.0, [0, 0, 0]),  # zero demand
            (2, 10, 10.1, 10.0, [0, 0, 0]),  # zero demand, right next to city 1
            (3, 10, 11.0, 10.0, [900, 900, 900]),  # the genuine neighbour
            (4, 10, -5.0, 0.0, [900, 900, 900]),  # close to (0, 0) but ~1,800 km away
        ]
    )
    legacy = _cluster_of(cluster_cities(cities, search, ClusteringConfig(floor=500)))
    fixed = _cluster_of(
        cluster_cities(cities, search, ClusteringConfig(floor=500, fix_zero_volume_centroid=True))
    )
    assert legacy[1] == legacy[2] == legacy[4]  # dragged towards (0, 0)
    assert fixed[1] == fixed[2] == fixed[3]  # joins the real neighbour
    assert fixed[4] == 4
