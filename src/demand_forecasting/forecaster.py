"""Train a production forecaster, persist it, and use it to forecast a month.

Artifacts are written as plain text (LightGBM model text, JSON, CSV) - deliberately **not**
pickle, which executes arbitrary code on load and is unsafe to distribute.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from . import __version__
from .config import MAX_LEAD, ModelConfig
from .features import (
    FEATURES,
    FittedComponents,
    HistoryStats,
    build_inference_frame,
    build_training_panel,
)
from .jalali import JalaliMonth
from .model import Calibration, StageBModel, ValidationReport, apply_guardrail, fit_stage_b
from .reveal_curve import RevealCurve

logger = logging.getLogger(__name__)

ARTIFACT_FILES = (
    "model.txt",
    "metadata.json",
    "reveal_per_cluster.csv",
    "reveal_global.csv",
    "reveal_dow_adjustment.csv",
    "history_overall.csv",
    "history_by_weekday.csv",
)


@dataclass
class TrainedForecaster:
    """Stage A + Stage B, fitted on all history before ``target`` starts."""

    model: StageBModel
    components: FittedComponents
    target: JalaliMonth
    validation: ValidationReport | None = None

    def forecast(
        self,
        visible: pd.DataFrame,
        holidays: pd.DataFrame,
        window: JalaliMonth | None = None,
        *,
        guardrail: bool = False,
    ) -> pd.DataFrame:
        """Predict final demand for every (cluster, check-in) of ``window`` (default: target).

        Returns ``cluster_code, checkin (date), predicted_demand (int)`` sorted by cluster and
        date. ``visible`` must include the searches logged so far for the window (i.e. be built
        from ``search_data.csv`` + ``evaluation.csv``).

        ``guardrail=True`` additionally clips Stage B to ``[0.7, 1.5] x`` Stage A on
        high-volume cells (off by default; it protects seasonal-spike months at a small cost
        elsewhere, see ``docs/EXPERIMENTS.md``).
        """
        window = window or self.target
        pairs = (
            visible[["cluster_code"]]
            .drop_duplicates()
            .merge(pd.DataFrame({"checkin": pd.date_range(window.start, window.end)}), how="cross")
        )
        cells = build_inference_frame(pairs, window.cutoff, visible, self.components, holidays)
        predicted = self.model.predict(cells)
        if guardrail:
            predicted = apply_guardrail(predicted, cells["stageA_estimate"].to_numpy())
        out = cells[["cluster_code", "checkin"]].copy()
        out["predicted_demand"] = predicted.round().astype(int)
        out["checkin"] = out["checkin"].dt.date
        return out.sort_values(["cluster_code", "checkin"]).reset_index(drop=True)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        """Write the artifacts into ``directory`` (created if needed)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.booster.save_model(str(directory / "model.txt"))

        reveal = self.components.reveal
        reveal.per_cluster.to_csv(directory / "reveal_per_cluster.csv")
        reveal.global_curve.to_csv(directory / "reveal_global.csv", header=["value"])
        reveal.dow_adjustment.to_csv(directory / "reveal_dow_adjustment.csv")
        self.components.history.overall.to_csv(directory / "history_overall.csv", index=False)
        self.components.history.by_weekday.to_csv(directory / "history_by_weekday.csv", index=False)

        metadata = {
            "package_version": __version__,
            "lightgbm_version": lgb.__version__,
            "target": str(self.target),
            "hist_end": self.components.hist_end.date().isoformat(),
            "features": self.model.features,
            "calibration": self.model.calibration.to_dict(),
            "max_lead": MAX_LEAD,
        }
        if self.validation is not None:
            metadata["validation"] = {
                "wape_stage_a": self.validation.wape_stage_a,
                "wape_stage_b": self.validation.wape_stage_b,
                "best_iteration": self.validation.best_iteration,
                "n_train": self.validation.n_train,
                "n_validation": self.validation.n_validation,
            }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("saved forecaster artifacts to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> TrainedForecaster:
        """Load artifacts written by :meth:`save`."""
        directory = Path(directory)
        missing = [f for f in ARTIFACT_FILES if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(f"{directory} is missing artifact files: {missing}")
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["features"] != FEATURES:
            raise ValueError(
                "artifact feature list differs from this package's FEATURES; "
                "retrain the model with the installed version"
            )

        def read(name: str, **kwargs) -> pd.DataFrame:
            return pd.read_csv(directory / name, float_precision="round_trip", **kwargs)

        per_cluster = read("reveal_per_cluster.csv", index_col=0)
        per_cluster.columns = per_cluster.columns.astype(int)
        dow_adjustment = read("reveal_dow_adjustment.csv", index_col=0)
        dow_adjustment.columns = dow_adjustment.columns.astype(int)
        global_curve = read("reveal_global.csv", index_col=0)["value"]
        for frame in (per_cluster, dow_adjustment):
            frame.columns.name = "lead"
        global_curve.index = global_curve.index.astype(int)
        global_curve.index.name = "lead"

        components = FittedComponents(
            hist_end=pd.Timestamp(metadata["hist_end"]),
            reveal=RevealCurve(per_cluster, global_curve, dow_adjustment),
            history=HistoryStats(read("history_overall.csv"), read("history_by_weekday.csv")),
        )
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        model = StageBModel(
            booster, Calibration.from_dict(metadata["calibration"]), list(metadata["features"])
        )
        return cls(model, components, JalaliMonth.parse(metadata["target"]))


def train_forecaster(
    visible: pd.DataFrame,
    holidays: pd.DataFrame,
    target: JalaliMonth,
    config: ModelConfig | None = None,
) -> TrainedForecaster:
    """Fit on every check-in before ``target`` starts, ready to forecast ``target``."""
    config = config or ModelConfig()
    components = FittedComponents.fit(visible, hist_end=target.start)
    panel = build_training_panel(visible, components, holidays, config.leads)
    logger.info("training panel: %s rows", f"{len(panel):,}")
    model, validation = fit_stage_b(panel, hist_end=target.start, config=config)
    return TrainedForecaster(model, components, target, validation)
