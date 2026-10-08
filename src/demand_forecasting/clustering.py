"""Merge sparse cities into clusters so every cluster has enough demand to forecast.

Tiny cities produce erratic, near-empty series; forecasting them individually adds noise
(including to the shared model) but aggregating too aggressively is penalised by the task, so
the rule is deliberately *minimal*:

    Starting from one cluster per city, repeatedly merge any cluster whose demand falls below
    ``floor`` in **at least one month** of the reference year into its **geographically nearest
    cluster of the same province**, until every cluster clears the floor in every month (or its
    province has nothing left to merge with).

Tracking the full per-month demand *vector* (not one scalar total) matters: a seasonal town
can look healthy on average yet be empty for a month, which an average threshold would miss.

Merged clusters sit at the demand-weighted centroid of their members. The original
implementation collapses that centroid to (0 deg, 0 deg) when two **zero-demand** clusters
merge, which later makes them "nearest" to whatever cluster lies closest to the Gulf of
Guinea. The default reproduces that behaviour bit-for-bit (so the shipped 148-cluster mapping
is reproducible); ``fix_zero_volume_centroid=True`` uses the plain mean of the coordinates
instead. See ``docs/EXPERIMENTS.md`` (#35) for its measured effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .jalali import jalali_parts

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class ClusteringConfig:
    floor: float = 500.0
    """Minimum searches a cluster must receive in *every* month of the reference year."""
    jalali_year: int | None = None
    """Reference Jalali year; default = the year of the latest check-in in the logs."""
    fix_zero_volume_centroid: bool = False
    """Use the mean of the coordinates (not (0, 0)) when merging zero-demand clusters."""


@dataclass
class _Group:
    members: list[int]
    vec: np.ndarray
    lat: float
    lon: float


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (broadcasts over numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def monthly_demand_vectors(
    cities: pd.DataFrame, search: pd.DataFrame, jalali_year: int | None = None
) -> tuple[np.ndarray, list[int], int]:
    """Searches per (city, Jalali month) of the reference year.

    Returns ``(matrix, months, year)`` with one row per city (in ``cities`` order) and one
    column per month that occurs in the reference year.
    """
    parts = jalali_parts(search["checkin"])
    if jalali_year is None:
        jalali_year = int(parts["jalali_year"].max())
    in_year = (parts["jalali_year"] == jalali_year).to_numpy()
    if not in_year.any():
        raise ValueError(f"no check-ins fall in Jalali year {jalali_year}")
    frame = pd.DataFrame(
        {
            "city_code": search["city_code"].to_numpy()[in_year],
            "jalali_month": parts["jalali_month"].to_numpy()[in_year],
            "search_count": search["search_count"].to_numpy()[in_year],
        }
    )
    months = sorted(frame["jalali_month"].unique().tolist())
    pivot = frame.groupby(["city_code", "jalali_month"])["search_count"].sum().unstack(fill_value=0)
    pivot = pivot.reindex(columns=months, fill_value=0).reindex(cities["city_code"], fill_value=0)
    return pivot.to_numpy(), months, jalali_year


def _merge_province(
    codes: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    vectors: np.ndarray,
    config: ClusteringConfig,
) -> dict[int, int]:
    """Agglomerate the cities of one province; returns ``{city_code: cluster_code}``."""
    groups: dict[int, _Group] = {
        int(code): _Group([int(code)], vec.astype(float), float(lat), float(lon))
        for code, lat, lon, vec in zip(codes, lats, lons, vectors, strict=True)
    }

    while len(groups) > 1:
        under = [gid for gid, group in groups.items() if group.vec.min() < config.floor]
        if not under:
            break
        gids = list(groups)
        position = {gid: i for i, gid in enumerate(gids)}
        all_lat = np.array([groups[g].lat for g in gids])
        all_lon = np.array([groups[g].lon for g in gids])

        best: tuple[int, int] | None = None
        best_distance = np.inf
        for gi in under:  # first minimal pair wins ties, exactly like a nested scan
            distance = haversine_km(groups[gi].lat, groups[gi].lon, all_lat, all_lon)
            distance = np.where(np.isnan(distance), np.inf, distance)
            distance[position[gi]] = np.inf
            j = int(np.argmin(distance))
            if distance[j] < best_distance:
                best_distance, best = float(distance[j]), (gi, gids[j])
        if best is None:
            break

        gi, gj = best
        a, b = groups[gi], groups[gj]
        vec = a.vec + b.vec
        total = vec.sum()
        weight_a, weight_b = a.vec.sum(), b.vec.sum()
        if total == 0 and config.fix_zero_volume_centroid:
            lat, lon = (a.lat + b.lat) / 2, (a.lon + b.lon) / 2
        else:
            denominator = max(total, 1e-9)
            lat = (a.lat * weight_a + b.lat * weight_b) / denominator
            lon = (a.lon * weight_a + b.lon * weight_b) / denominator
        groups[gi] = _Group(a.members + b.members, vec, lat, lon)
        del groups[gj]

    return {member: gid for gid, group in groups.items() for member in group.members}


def cluster_cities(
    cities: pd.DataFrame, search: pd.DataFrame, config: ClusteringConfig | None = None
) -> pd.DataFrame:
    """Build the city -> cluster mapping.

    Parameters
    ----------
    cities:
        ``city_code, province_code, lat, long`` (one row per city).
    search:
        Raw search logs (only the check-in month and city totals are used).

    Returns
    -------
    DataFrame with ``city_code, cluster_code``; the cluster code is the code of one of its
    member cities.
    """
    config = config or ClusteringConfig()
    vectors, months, year = monthly_demand_vectors(cities, search, config.jalali_year)
    cities = cities.reset_index(drop=True)
    mapping = {int(code): int(code) for code in cities["city_code"]}

    for province, group in cities.groupby("province_code", sort=False):
        if len(group) <= 1:
            continue
        rows = group.index.to_numpy()
        mapping.update(
            _merge_province(
                group["city_code"].to_numpy(),
                group["lat"].to_numpy(),
                group["long"].to_numpy(),
                vectors[rows],
                config,
            )
        )
        logger.debug("province %s: %d cities", province, len(group))

    result = pd.DataFrame(
        {"city_code": cities["city_code"], "cluster_code": cities["city_code"].map(mapping)}
    )
    per_cluster = (
        pd.DataFrame(vectors, index=result["cluster_code"].to_numpy()).groupby(level=0).sum()
    )
    below = int((per_cluster.min(axis=1) < config.floor).sum())
    logger.info(
        "floor=%g/month over %d months of Jalali %d: %d cities -> %d clusters, "
        "%d below the floor (their province has nothing left to merge with)",
        config.floor,
        len(months),
        year,
        len(cities),
        result["cluster_code"].nunique(),
        below,
    )
    return result
