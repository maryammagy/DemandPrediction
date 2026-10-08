"""Feature construction for Stage B.

Two row-sets are built by the *same* code path so that training and inference see identical
feature semantics:

* :func:`build_training_panel` - **replays history**: every historical (cluster, check-in) is
  observed at many simulated lead times, with Stage A's estimate and the true final demand
  attached. The model therefore learns in exactly the partial-information regime it faces
  when forecasting.
* :func:`build_inference_frame` - one row per (cluster, future check-in) as of a fixed cutoff.

**No-leakage rule.** Every statistic (reveal curve, per-cluster history, prior-year demand) is
computed from check-ins strictly before ``hist_end`` - the first day of the window being
scored - so the model never sees the future it is asked to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar_features import add_calendar_features
from .config import MAX_LEAD
from .data import FINAL_DEMAND_COLUMN, LEAD_COLUMNS
from .reveal_curve import RevealCurve

FEATURES: list[str] = [
    # what is known about this check-in right now
    "lead",
    "partial_observed_log",
    "stageA_estimate_log",
    # calendar
    "month",
    "dow",
    "jalali_month",
    "jalali_week",
    "week_of_month",
    "is_holiday",
    "is_weekend",
    "is_day_off",
    "is_bridge_day",
    "in_holiday_block",
    "days_to_nearest_holiday",
    "nearest_block_length",
    # how big / how variable this cluster usually is (overall and for this weekday)
    "cluster_hist_mean_log",
    "cluster_hist_std_log",
    "cluster_dow_hist_mean_log",
    "cluster_dow_hist_std_log",
    # same date one year earlier
    "prior_year_demand_log",
]
"""Exact feature list and order the Stage-B model expects."""


@dataclass
class HistoryStats:
    """Per-cluster demand level/volatility, overall and per weekday (raw, not log, scale)."""

    overall: pd.DataFrame  # cluster_code, cluster_hist_mean, cluster_hist_std
    by_weekday: pd.DataFrame  # cluster_code, dow_raw, cluster_dow_hist_mean, cluster_dow_hist_std

    @classmethod
    def fit(cls, visible: pd.DataFrame, hist_end: pd.Timestamp) -> HistoryStats:
        hist = visible[visible["checkin"] < hist_end]

        overall = (
            hist.groupby("cluster_code")[FINAL_DEMAND_COLUMN].agg(["mean", "std"]).reset_index()
        )
        overall.columns = ["cluster_code", "cluster_hist_mean", "cluster_hist_std"]
        overall["cluster_hist_std"] = overall["cluster_hist_std"].fillna(0)

        # Weekday matters in *level*, not just in reveal timing: Wednesday/Thursday sit well
        # above Monday/Tuesday for most clusters, which a single pooled mean would blur.
        weekday = hist["checkin"].dt.dayofweek.to_numpy()
        by_weekday = (
            hist.groupby(["cluster_code", weekday])[FINAL_DEMAND_COLUMN]
            .agg(["mean", "std"])
            .reset_index()
        )
        by_weekday.columns = [
            "cluster_code",
            "dow_raw",
            "cluster_dow_hist_mean",
            "cluster_dow_hist_std",
        ]
        by_weekday["cluster_dow_hist_std"] = by_weekday["cluster_dow_hist_std"].fillna(0)
        return cls(overall, by_weekday)

    def attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the statistics onto ``df`` and add their ``log1p`` versions."""
        out = df.merge(self.overall, on="cluster_code", how="left")
        out["cluster_hist_mean"] = out["cluster_hist_mean"].fillna(0)
        out["cluster_hist_std"] = out["cluster_hist_std"].fillna(0)
        out["cluster_hist_mean_log"] = np.log1p(out["cluster_hist_mean"])
        out["cluster_hist_std_log"] = np.log1p(out["cluster_hist_std"])

        out["dow_raw"] = out["checkin"].dt.dayofweek
        out = out.merge(self.by_weekday, on=["cluster_code", "dow_raw"], how="left")
        # sparse (cluster, weekday) cells fall back to the cluster-wide statistic
        out["cluster_dow_hist_mean"] = out["cluster_dow_hist_mean"].fillna(out["cluster_hist_mean"])
        out["cluster_dow_hist_std"] = out["cluster_dow_hist_std"].fillna(out["cluster_hist_std"])
        out["cluster_dow_hist_mean_log"] = np.log1p(out["cluster_dow_hist_mean"])
        out["cluster_dow_hist_std_log"] = np.log1p(out["cluster_dow_hist_std"])
        return out.drop(columns=["dow_raw"])


@dataclass
class FittedComponents:
    """Everything learned from history before ``hist_end`` (Stage A + cluster statistics)."""

    hist_end: pd.Timestamp
    reveal: RevealCurve
    history: HistoryStats

    @classmethod
    def fit(cls, visible: pd.DataFrame, hist_end: pd.Timestamp) -> FittedComponents:
        return cls(
            hist_end=pd.Timestamp(hist_end),
            reveal=RevealCurve.fit(visible, hist_end),
            history=HistoryStats.fit(visible, hist_end),
        )


def attach_prior_year(
    df: pd.DataFrame, visible: pd.DataFrame, hist_end: pd.Timestamp, tol_days: int = 6
) -> pd.DataFrame:
    """Add ``prior_year_demand``: the cluster's final demand ~365 days earlier.

    The nearest historical check-in within ``tol_days`` of (check-in - 365 days) is used;
    0 when none exists (first year of data). Only dates before ``hist_end`` are eligible.
    """
    history = visible.loc[
        visible["checkin"] < hist_end, ["cluster_code", "checkin", FINAL_DEMAND_COLUMN]
    ].rename(columns={"checkin": "prior_checkin", FINAL_DEMAND_COLUMN: "prior_year_demand"})

    out = df.copy()
    out["target_prior_date"] = out["checkin"] - pd.Timedelta(days=365)
    out = out.sort_values("target_prior_date")
    merged = pd.merge_asof(
        out,
        history.sort_values("prior_checkin"),
        left_on="target_prior_date",
        right_on="prior_checkin",
        by="cluster_code",
        direction="nearest",
        tolerance=pd.Timedelta(days=tol_days),
    )
    merged["prior_year_demand"] = merged["prior_year_demand"].fillna(0)
    return merged.drop(columns=["target_prior_date", "prior_checkin"])


def attach_features(
    df: pd.DataFrame,
    components: FittedComponents,
    visible: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Turn rows carrying ``partial_observed`` / ``stageA_estimate`` into model features."""
    out = df.copy()
    out["partial_observed_log"] = np.log1p(out["partial_observed"])
    out["stageA_estimate_log"] = np.log1p(out["stageA_estimate"])
    out = add_calendar_features(out, "checkin", holidays)
    out = components.history.attach(out)
    out = attach_prior_year(out, visible, components.hist_end)
    out["prior_year_demand_log"] = np.log1p(out["prior_year_demand"])
    return out


def build_training_panel(
    visible: pd.DataFrame,
    components: FittedComponents,
    holidays: pd.DataFrame,
    leads: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Replay history at simulated cutoffs to build the Stage-B training set.

    For every lead ``L`` in ``leads`` and every (cluster, check-in) before ``hist_end`` the row
    records what an observer ``L`` days before check-in would have seen (``partial_observed``),
    Stage A's resulting estimate, and the true final demand (``target``).
    """
    hist = visible[visible["checkin"] < components.hist_end]
    clusters = hist["cluster_code"].to_numpy()
    checkins = hist["checkin"].to_numpy()
    weekdays = hist["checkin"].dt.dayofweek.to_numpy()
    target = hist[FINAL_DEMAND_COLUMN].to_numpy()

    chunks = []
    for lead in leads:
        partial = hist[LEAD_COLUMNS[lead]].to_numpy()
        fraction = components.reveal.fraction(clusters, np.full(len(hist), lead), weekdays)
        chunks.append(
            pd.DataFrame(
                {
                    "cluster_code": clusters,
                    "checkin": checkins,
                    "lead": lead,
                    "partial_observed": partial,
                    "stageA_estimate": partial / fraction,
                    "target": target,
                }
            )
        )
    panel = pd.concat(chunks, ignore_index=True)
    panel["target_log"] = np.log1p(panel["target"])
    return attach_features(panel, components, visible, holidays)


def build_inference_frame(
    pairs: pd.DataFrame,
    cutoff: pd.Timestamp,
    visible: pd.DataFrame,
    components: FittedComponents,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Feature rows for future (cluster, check-in) ``pairs`` as observed at ``cutoff``.

    ``lead = checkin - cutoff`` is the forecast horizon; ``partial_observed`` is the demand
    visible at that lead, i.e. the searches logged up to and including ``cutoff``.
    """
    df = pairs.merge(visible, on=["cluster_code", "checkin"], how="left")
    df[LEAD_COLUMNS] = df[LEAD_COLUMNS].fillna(0)  # no log rows == zero searches
    df["lead"] = (df["checkin"] - pd.Timestamp(cutoff)).dt.days
    df = df[df["lead"].between(1, MAX_LEAD)].copy()

    matrix = df[LEAD_COLUMNS].to_numpy()
    df["partial_observed"] = matrix[np.arange(len(df)), df["lead"].to_numpy()]
    fraction = components.reveal.fraction(
        df["cluster_code"].to_numpy(), df["lead"].to_numpy(), df["checkin"].dt.dayofweek.to_numpy()
    )
    df["stageA_estimate"] = df["partial_observed"] / fraction
    return attach_features(df, components, visible, holidays)
