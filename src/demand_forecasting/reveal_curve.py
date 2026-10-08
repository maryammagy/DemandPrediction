"""Stage A - the *reveal curve*.

For a future check-in date only the searches logged so far are known, so its demand is
**right-censored**. If, historically, a fraction ``r(L)`` of a check-in's eventual demand has
been logged by ``L`` days before arrival, then::

    demand_hat = observed_so_far / r(L)

``r(L)`` is estimated as the median (over history) of ``visible_at_lead_L / final_demand``,

* per cluster, falling back to the pooled curve for clusters without history, and
* multiplied by a pooled per-weekday factor, because Thursday/Friday (Iran's weekend)
  check-ins are booked further ahead, so a fixed lead reveals a smaller share of them.

Only check-ins strictly before ``hist_end`` are used, so nothing leaks from the window being
forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MAX_LEAD, REVEAL_FRACTION_FLOOR
from .data import FINAL_DEMAND_COLUMN, LEAD_COLUMNS

_LEADS = np.arange(MAX_LEAD + 1)
_DOW_ADJUSTMENT_BOUNDS = (0.3, 3.0)


@dataclass
class RevealCurve:
    """Fitted reveal fractions.

    Attributes
    ----------
    per_cluster:
        Median revealed fraction per cluster; index ``cluster_code``, columns ``0..MAX_LEAD``.
    global_curve:
        Pooled median over all clusters; index ``0..MAX_LEAD``.
    dow_adjustment:
        Multiplicative weekday correction (index = pandas weekday, Monday = 0), clipped to
        ``[0.3, 3.0]``; columns ``0..MAX_LEAD``.
    """

    per_cluster: pd.DataFrame
    global_curve: pd.Series
    dow_adjustment: pd.DataFrame

    @classmethod
    def fit(cls, visible: pd.DataFrame, hist_end: pd.Timestamp) -> RevealCurve:
        """Fit on the check-ins strictly before ``hist_end`` that received any demand."""
        hist = visible[visible["checkin"] < hist_end]
        total = hist[FINAL_DEMAND_COLUMN]
        keep = total > 0
        hist, total = hist[keep], total[keep]

        fraction = hist[LEAD_COLUMNS].div(total, axis=0)
        fraction.columns = _LEADS
        per_cluster = fraction.groupby(hist["cluster_code"].to_numpy()).median()
        per_cluster.index.name = "cluster_code"
        global_curve = fraction.median()

        weekday = hist["checkin"].dt.dayofweek.to_numpy()
        by_weekday = fraction.groupby(weekday).median()
        by_weekday.index.name = "dow"
        low, high = _DOW_ADJUSTMENT_BOUNDS
        dow_adjustment = by_weekday.div(global_curve, axis=1).clip(lower=low, upper=high)

        per_cluster = per_cluster.reindex(columns=_LEADS)
        global_curve.index.name = "lead"
        per_cluster.columns.name = dow_adjustment.columns.name = "lead"
        return cls(per_cluster, global_curve, dow_adjustment)

    def fraction(
        self,
        cluster_codes: np.ndarray,
        leads: np.ndarray,
        weekdays: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reveal fraction for each row (vectorised).

        ``weekdays`` are pandas weekdays (Monday = 0); omit them to skip the weekday factor.
        """
        leads = np.clip(np.asarray(leads), 0, MAX_LEAD).astype(int)
        slot = self.per_cluster.index.get_indexer(cluster_codes)
        table = self.per_cluster.to_numpy(dtype=float)
        values = np.where(slot >= 0, table[np.maximum(slot, 0), leads], np.nan)
        values = np.where(np.isnan(values), self.global_curve.to_numpy(dtype=float)[leads], values)

        if weekdays is not None:
            adjustment = self.dow_adjustment.reindex(index=range(7), columns=_LEADS)
            adjustment = np.nan_to_num(adjustment.to_numpy(dtype=float), nan=1.0)
            values = values * adjustment[np.asarray(weekdays).astype(int), leads]
        return np.clip(values, REVEAL_FRACTION_FLOOR, None)

    def naive_fraction(self, leads: np.ndarray) -> np.ndarray:
        """Pooled curve only (no cluster or weekday effects) - the naive baseline."""
        leads = np.clip(np.asarray(leads), 0, MAX_LEAD).astype(int)
        return np.clip(self.global_curve.to_numpy(dtype=float)[leads], REVEAL_FRACTION_FLOOR, 1)
