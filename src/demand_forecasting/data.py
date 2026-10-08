"""Loading the raw search logs and reshaping them for forecasting.

Vocabulary used throughout the package:

``lead``
    Days between a search (``log_date``) and the intended stay (``checkin``).
``demand``
    Total searches for a (cluster, check-in) pair over *all* log dates - the target.
``visible demand at lead L``
    Searches with lead >= L, i.e. what an observer standing L days before the check-in
    has seen so far. ``visible_at_lead_0`` is the final demand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import MAX_LEAD

logger = logging.getLogger(__name__)

SEARCH_COLUMNS = ["log_date", "city_code", "checkin", "search_count"]
LEAD_COLUMNS = [f"visible_at_lead_{lead}" for lead in range(MAX_LEAD + 1)]
FINAL_DEMAND_COLUMN = LEAD_COLUMNS[0]


def read_search_logs(path: str | Path) -> pd.DataFrame:
    """Read one search-log CSV (``log_date, city_code, checkin, search_count``)."""
    df = pd.read_csv(path, usecols=SEARCH_COLUMNS, parse_dates=["log_date", "checkin"])
    missing = set(SEARCH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df[SEARCH_COLUMNS].isna().any().any():
        raise ValueError(f"{path}: null values in {SEARCH_COLUMNS}")
    if (df["search_count"] < 0).any():
        raise ValueError(f"{path}: negative search_count")
    return df


def load_search_data(data_dir: str | Path, *, include_evaluation: bool = False) -> pd.DataFrame:
    """Load ``search_data.csv`` and, optionally, the partial ``evaluation.csv``.

    The evaluation file holds the searches logged so far for the *target* month; it is only
    concatenated when producing the real forecast, never when backtesting.
    """
    data_dir = Path(data_dir)
    frames = [read_search_logs(data_dir / "search_data.csv")]
    if include_evaluation:
        frames.append(read_search_logs(data_dir / "evaluation.csv"))
    search = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info("loaded %s search rows from %s", f"{len(search):,}", data_dir)
    return search


def load_cities(data_dir: str | Path) -> pd.DataFrame:
    """Static city metadata: ``city_code, province_code, lat, long``."""
    return pd.read_csv(Path(data_dir) / "cities.csv")


def load_clusters(path: str | Path) -> pd.DataFrame:
    """City -> cluster mapping: ``city_code, cluster_code``."""
    return pd.read_csv(path)


def aggregate_to_clusters(search: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Sum city-level searches into cluster-level searches per (log date, check-in)."""
    merged = search.merge(clusters, on="city_code", how="left")
    unmapped = merged.loc[merged["cluster_code"].isna(), "city_code"].unique()
    if len(unmapped):
        raise ValueError(
            f"{len(unmapped)} city codes have no cluster, e.g. {unmapped[:5].tolist()}"
        )
    return merged.groupby(["cluster_code", "log_date", "checkin"], as_index=False)[
        "search_count"
    ].sum()


def build_visible_demand(cluster_search: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Visible demand at every lead time, one row per (cluster, check-in).

    Row (c, t) holds ``visible_at_lead_L`` for L = 0..MAX_LEAD, the reverse cumulative sum of
    the searches logged at each lead. The frame is a *zero-filled grid*: every cluster in
    ``clusters`` crossed with every distinct check-in date seen in the logs, because a
    missing log row means zero searches.
    """
    cs = cluster_search.copy()
    cs["lead"] = (cs["checkin"] - cs["log_date"]).dt.days
    cs = cs[cs["lead"].between(0, MAX_LEAD)]

    per_lead = cs.groupby(["cluster_code", "checkin", "lead"])["search_count"].sum().reset_index()
    pivot = per_lead.pivot_table(
        index=["cluster_code", "checkin"], columns="lead", values="search_count", fill_value=0
    )
    pivot = pivot.reindex(columns=range(0, MAX_LEAD + 1), fill_value=0)
    # reverse cumulative sum: column L = searches with lead >= L
    visible = pivot.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    visible.columns = LEAD_COLUMNS
    visible = visible.reset_index()

    grid = pd.MultiIndex.from_product(
        [clusters["cluster_code"].unique(), cluster_search["checkin"].unique()],
        names=["cluster_code", "checkin"],
    ).to_frame(index=False)
    visible = grid.merge(visible, on=["cluster_code", "checkin"], how="left")
    visible[LEAD_COLUMNS] = visible[LEAD_COLUMNS].fillna(0)
    return visible
