"""Stage B - LightGBM correction of Stage A, plus calibration and a guardrail.

The regressor is trained on ``log1p(final demand)`` with an L1 loss. That is robust and
scale-free, but it predicts close to the conditional *median*, which on this right-skewed
target sits below the conditional *mean* that WAPE rewards. :class:`Calibration` therefore
rescales predictions per forecast-horizon bucket with a factor estimated on a held-out fold
(never on the scored window) and **floored at 1.0**: because median <= mean the true correction
is always an up-scaling, so any estimate below 1 is fold noise, not signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import LEAD_BUCKET_EDGES, ModelConfig
from .features import FEATURES
from .metrics import wape

logger = logging.getLogger(__name__)

# Guardrail (experiment #15): on high-volume cells, never let Stage B stray far from Stage A.
GUARD_LOW = 0.7
GUARD_HIGH = 1.5
GUARD_MIN_SCALE = 1000


def bucket_label(low: int, high: int) -> str:
    return f"({low}, {high}]"


def bucket_index(leads: np.ndarray, edges: tuple[int, ...] = LEAD_BUCKET_EDGES) -> np.ndarray:
    """Index of the right-closed bucket (edges[i], edges[i+1]] each lead falls in, else -1."""
    leads = np.asarray(leads)
    idx = np.searchsorted(np.asarray(edges), leads, side="left") - 1
    return np.where((idx < 0) | (idx >= len(edges) - 1), -1, idx)


@dataclass
class Calibration:
    """Multiplicative bias correction: one global factor and one factor per lead bucket."""

    global_factor: float = 1.0
    by_bucket: dict[str, float] = field(default_factory=dict)
    edges: tuple[int, ...] = LEAD_BUCKET_EDGES

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        leads: np.ndarray,
        edges: tuple[int, ...] = LEAD_BUCKET_EDGES,
    ) -> Calibration:
        """Ratio of summed truth to summed prediction, floored at 1.0 (see module docs)."""
        y_true = pd.Series(np.asarray(y_true, dtype=float))
        y_pred = pd.Series(np.asarray(y_pred, dtype=float))
        global_factor = max(y_true.sum() / max(y_pred.sum(), 1e-9), 1.0)

        buckets = pd.cut(pd.Series(np.asarray(leads)), list(edges))
        true_sum = y_true.groupby(buckets.values, observed=True).sum()
        pred_sum = y_pred.groupby(buckets.values, observed=True).sum().clip(lower=1e-9)
        factors = (true_sum / pred_sum).clip(lower=1.0)
        by_bucket = {str(interval): float(value) for interval, value in factors.items()}
        return cls(float(global_factor), by_bucket, tuple(edges))

    def factors(self, leads: np.ndarray) -> np.ndarray:
        """Per-row factor; leads outside every bucket (or without data) use the global one."""
        idx = bucket_index(leads, self.edges)
        table = np.array(
            [
                self.by_bucket.get(bucket_label(lo, hi), self.global_factor)
                for lo, hi in zip(self.edges[:-1], self.edges[1:], strict=True)
            ]
            + [self.global_factor]
        )
        return table[idx]  # idx == -1 selects the trailing global factor

    def to_dict(self) -> dict:
        return {
            "cal_global": self.global_factor,
            "cal_by_lead_bucket": self.by_bucket,
            "lead_bucket_edges": list(self.edges),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Calibration:
        return cls(
            float(payload["cal_global"]),
            {k: float(v) for k, v in payload["cal_by_lead_bucket"].items()},
            tuple(payload["lead_bucket_edges"]),
        )


def apply_guardrail(
    stage_b: np.ndarray,
    stage_a: np.ndarray,
    *,
    low: float = GUARD_LOW,
    high: float = GUARD_HIGH,
    min_scale: float = GUARD_MIN_SCALE,
) -> np.ndarray:
    """Clip Stage B to ``[low, high] x`` Stage A on high-volume cells (Stage A >= ``min_scale``).

    Restricting the clip to large cells matters: applied everywhere it mostly clips tiny,
    noisy cells and makes WAPE worse. Reported by the backtests; not applied to the shipped
    forecast.
    """
    stage_b, stage_a = np.asarray(stage_b, dtype=float), np.asarray(stage_a, dtype=float)
    clipped = np.clip(stage_b, low * stage_a, high * stage_a)
    return np.where(stage_a >= min_scale, clipped, stage_b)


@dataclass
class StageBModel:
    """A trained booster together with its feature list and calibration."""

    booster: lgb.Booster
    calibration: Calibration
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        """Uncalibrated demand prediction (back-transformed from log1p, floored at 0)."""
        return np.expm1(self.booster.predict(frame[self.features])).clip(min=0)

    def predict(self, frame: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        raw = self.predict_raw(frame)
        return raw * self.calibration.factors(frame["lead"].to_numpy()) if calibrated else raw


@dataclass
class ValidationReport:
    """Diagnostics from the early-stopping fold."""

    wape_stage_a: float
    wape_stage_b: float
    best_iteration: int
    n_train: int
    n_validation: int
    feature_importance: pd.Series


def fit_stage_b(
    panel: pd.DataFrame, hist_end: pd.Timestamp, config: ModelConfig | None = None
) -> tuple[StageBModel, ValidationReport]:
    """Train Stage B on a training panel and fit its calibration on the validation fold.

    The validation fold is the last ``config.validation_days`` days of check-ins before
    ``hist_end``; it drives early stopping *and* the calibration factors.
    """
    config = config or ModelConfig()
    is_val = panel["checkin"] >= hist_end - pd.Timedelta(days=config.validation_days)
    train, val = panel[~is_val], panel[is_val]
    logger.info("stage B: %s train rows, %s validation rows", f"{len(train):,}", f"{len(val):,}")
    if train.empty or val.empty:
        raise ValueError("not enough history to build both a training and a validation fold")

    x_train, y_train = train[FEATURES], train["target_log"]
    x_val, y_val = val[FEATURES], val["target_log"]
    if config.sample_weighting == "sqrt":
        w_train, w_val = np.sqrt(np.expm1(y_train)), np.sqrt(np.expm1(y_val))
    else:
        w_train = w_val = None

    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train)
    val_set = lgb.Dataset(x_val, label=y_val, weight=w_val, reference=train_set)
    booster = lgb.train(
        config.lgbm_params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_pred = np.expm1(booster.predict(x_val)).clip(min=0)
    val_true = np.expm1(y_val.to_numpy())
    calibration = Calibration.fit(val_true, val_pred, val["lead"].to_numpy())

    report = ValidationReport(
        wape_stage_a=wape(val_true, val["stageA_estimate"].to_numpy()),
        wape_stage_b=wape(val_true, val_pred),
        best_iteration=int(booster.best_iteration),
        n_train=len(train),
        n_validation=len(val),
        feature_importance=pd.Series(
            booster.feature_importance(importance_type="gain"), index=FEATURES
        ),
    )
    logger.info(
        "stage B: best iteration %d, validation WAPE stage A %.4f -> stage B %.4f",
        report.best_iteration,
        report.wape_stage_a,
        report.wape_stage_b,
    )
    return StageBModel(booster, calibration), report
