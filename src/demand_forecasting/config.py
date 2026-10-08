"""Constants and configuration objects shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_LEAD = 59
"""Longest lead time (days between log date and check-in) present in the search logs."""

LEAD_BUCKET_EDGES: tuple[int, ...] = (0, 3, 7, 14, 21, 30)
"""Right-closed forecast-horizon buckets used for calibration: (0,3], (3,7], ... (21,30]."""

TRAIN_LEADS: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 13, 16, 20, 22, 25, 27, 30)
"""Simulated "days before check-in" cutoffs replayed to build the Stage-B training panel."""

VALIDATION_DAYS = 60
"""The last N days of check-ins before the forecast window form the early-stopping fold."""

REVEAL_FRACTION_FLOOR = 0.02
"""Lower bound on the reveal fraction so Stage A never divides by (almost) zero."""

JALALI_MONTH_NAMES: tuple[str, ...] = (
    "Farvardin",
    "Ordibehesht",
    "Khordad",
    "Tir",
    "Mordad",
    "Shahrivar",
    "Mehr",
    "Aban",
    "Azar",
    "Dey",
    "Bahman",
    "Esfand",
)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",  # MAE on log1p(demand) -> robust, scale-free
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "seed": 42,
    "deterministic": True,  # reproducible trees ...
    "force_row_wise": True,  # ... (LightGBM recommends pairing these two)
    "verbosity": -1,
}


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of the Stage-B model and its training-panel construction."""

    leads: tuple[int, ...] = TRAIN_LEADS
    validation_days: int = VALIDATION_DAYS
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    sample_weighting: str = "none"
    """``"none"`` (default) or ``"sqrt"`` - weight each row by sqrt(true demand).

    The latter pushes the loss towards WAPE (demand-weighted) and roughly halves the error on
    seasonal-spike months, but costs accuracy elsewhere (see ``docs/EXPERIMENTS.md``, #21)."""

    def __post_init__(self) -> None:
        if self.sample_weighting not in {"none", "sqrt"}:
            raise ValueError("sample_weighting must be 'none' or 'sqrt'")
