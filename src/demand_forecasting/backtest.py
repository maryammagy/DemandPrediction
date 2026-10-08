"""Rolling-origin backtesting under the *same* partial-information regime as production.

To score month ``M`` we pretend today is the last day of month ``M-1``:

1. fit Stage A, the cluster statistics and Stage B using only check-ins before ``M``;
2. observe, for each check-in in ``M``, only the searches logged up to that cutoff;
3. forecast the final demand and compare with what was eventually logged.

A window is only scorable when the logs fully resolve it (every check-in in ``M`` is already
in the data), which is why the hold-out windows are *past* months.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .config import ModelConfig
from .data import FINAL_DEMAND_COLUMN
from .features import FittedComponents, build_inference_frame, build_training_panel
from .jalali import JalaliMonth
from .metrics import wape
from .model import ValidationReport, apply_guardrail, fit_stage_b

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS: tuple[JalaliMonth, ...] = (
    JalaliMonth(1403, 9),  # Azar 1403 - same calendar month, one year before the target
    JalaliMonth(1404, 5),  # Mordad 1404
    JalaliMonth(1404, 6),  # Shahrivar 1404 - seasonal-spike month (the hard case)
    JalaliMonth(1404, 8),  # Aban 1404 - the most recent fully observed month
)

PREDICTION_COLUMNS = {
    "naive": "pred_naive",
    "stage_a": "pred_stage_a",
    "stage_b": "pred_stage_b",
    "stage_b_guarded": "pred_stage_b_guarded",
    "stage_b_cal_global": "pred_stage_b_cal_global",
    "stage_b_cal_lead": "pred_stage_b_cal_lead",
    "stage_b_cal_guarded": "pred_stage_b_cal_guarded",
}


@dataclass
class BacktestResult:
    """Metrics and per-cell predictions of one backtest window."""

    window: JalaliMonth
    cutoff: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
    metrics: dict[str, float]
    frame: pd.DataFrame
    validation: ValidationReport
    calibration_global: float

    @property
    def label(self) -> str:
        return self.window.label

    def summary_row(self) -> dict[str, object]:
        return {
            "window": self.label,
            "cells": len(self.frame),
            **{k: round(v, 4) for k, v in self.metrics.items()},
        }


def run_backtest(
    visible: pd.DataFrame,
    holidays: pd.DataFrame,
    window: JalaliMonth,
    config: ModelConfig | None = None,
) -> BacktestResult:
    """Train on everything before ``window`` and score the window.

    Parameters
    ----------
    visible:
        Output of :func:`demand_forecasting.data.build_visible_demand` (built from
        ``search_data.csv`` only).
    """
    config = config or ModelConfig()
    start, end, cutoff = window.start, window.end, window.cutoff
    if end > visible["checkin"].max():
        raise ValueError(
            f"{window.label} ends {end.date()} but the logs only resolve check-ins up to "
            f"{visible['checkin'].max().date()}; it cannot be scored"
        )
    logger.info(
        "backtest %s: cutoff %s, window %s -> %s",
        window.label,
        cutoff.date(),
        start.date(),
        end.date(),
    )

    components = FittedComponents.fit(visible, hist_end=start)
    panel = build_training_panel(visible, components, holidays, config.leads)
    logger.info("backtest %s: %s synthetic training rows", window.label, f"{len(panel):,}")
    model, validation = fit_stage_b(panel, hist_end=start, config=config)

    pairs = visible[["cluster_code"]].drop_duplicates()
    pairs = pairs.merge(pd.DataFrame({"checkin": pd.date_range(start, end)}), how="cross")
    cells = build_inference_frame(pairs, cutoff, visible, components, holidays)
    cells["true_total"] = cells[FINAL_DEMAND_COLUMN]

    raw = model.predict_raw(cells)
    cal = model.calibration
    lead = cells["lead"].to_numpy()
    stage_a = cells["stageA_estimate"].to_numpy()
    calibrated = raw * cal.factors(lead)
    predictions = {
        "naive": cells["partial_observed"].to_numpy()
        / components.reveal.naive_fraction(lead),  # pooled curve only
        "stage_a": stage_a,
        "stage_b": raw,
        "stage_b_guarded": apply_guardrail(raw, stage_a),
        "stage_b_cal_global": raw * cal.global_factor,
        "stage_b_cal_lead": calibrated,  # the shipped configuration
        "stage_b_cal_guarded": apply_guardrail(calibrated, stage_a),
    }

    truth = cells["true_total"].to_numpy()
    metrics = {f"wape_{name}": wape(truth, pred) for name, pred in predictions.items()}
    metrics["wape_val_stage_a"] = validation.wape_stage_a
    metrics["wape_val_stage_b"] = validation.wape_stage_b

    frame = cells[["cluster_code", "checkin", "lead", "partial_observed", "true_total"]].copy()
    for name, column in PREDICTION_COLUMNS.items():
        frame[column] = predictions[name]
    frame = frame.reset_index(drop=True)

    logger.info(
        "backtest %s: WAPE naive %.4f | stage A %.4f | stage B %.4f | calibrated %.4f",
        window.label,
        metrics["wape_naive"],
        metrics["wape_stage_a"],
        metrics["wape_stage_b"],
        metrics["wape_stage_b_cal_lead"],
    )
    return BacktestResult(
        window=window,
        cutoff=cutoff,
        start=start,
        end=end,
        metrics=metrics,
        frame=frame,
        validation=validation,
        calibration_global=cal.global_factor,
    )


def run_backtests(
    visible: pd.DataFrame,
    holidays: pd.DataFrame,
    windows: tuple[JalaliMonth, ...] | list[JalaliMonth] = DEFAULT_WINDOWS,
    config: ModelConfig | None = None,
) -> list[BacktestResult]:
    return [run_backtest(visible, holidays, window, config) for window in windows]


def summarize(results: list[BacktestResult]) -> pd.DataFrame:
    """One row per window with the WAPE of every method."""
    return pd.DataFrame([r.summary_row() for r in results])


def lead_bucket_wape(result: BacktestResult, column: str = "pred_stage_b_cal_lead") -> pd.DataFrame:
    """WAPE and signed bias of ``column`` per forecast-horizon bucket (error vs. lead)."""
    frame = result.frame
    buckets = pd.cut(frame["lead"], [0, 3, 7, 14, 21, 30])
    rows = []
    for bucket, part in frame.groupby(buckets, observed=True):
        truth, pred = part["true_total"].to_numpy(), part[column].to_numpy()
        rows.append(
            {
                "lead_bucket": str(bucket),
                "demand_share": truth.sum() / frame["true_total"].sum(),
                "wape": wape(truth, pred),
                "bias": (pred.sum() - truth.sum()) / max(truth.sum(), 1e-9),
            }
        )
    return pd.DataFrame(rows)
