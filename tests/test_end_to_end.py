"""End-to-end tests on a small synthetic dataset (a few seconds each)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.backtest import lead_bucket_wape, run_backtest, summarize
from demand_forecasting.cli import main
from demand_forecasting.data import LEAD_COLUMNS
from demand_forecasting.features import build_inference_frame
from demand_forecasting.forecaster import TrainedForecaster, train_forecaster
from demand_forecasting.jalali import JalaliMonth
from demand_forecasting.metrics import wape
from demand_forecasting.model import GUARD_HIGH, GUARD_LOW, GUARD_MIN_SCALE

pytestmark = pytest.mark.slow

ABAN = JalaliMonth(1404, 8)  # a past, fully observed month -> can be backtested
AZAR = JalaliMonth(1404, 9)  # the target month: only partially observed


@pytest.fixture(scope="module")
def trained(visible_with_target, holidays, fast_model):
    return train_forecaster(visible_with_target, holidays, AZAR, fast_model)


def test_backtest_beats_the_naive_baseline(visible, holidays, fast_model):
    result = run_backtest(visible, holidays, ABAN, fast_model)
    n_clusters = visible["cluster_code"].nunique()

    assert len(result.frame) == n_clusters * 30
    assert result.frame["true_total"].sum() > 0
    metrics = result.metrics
    assert metrics["wape_stage_a"] < metrics["wape_naive"]  # per-cluster + weekday curve helps
    assert metrics["wape_stage_b_cal_lead"] < metrics["wape_naive"]
    for column in result.frame.filter(like="pred_").columns:
        assert (result.frame[column] >= 0).all()
    assert np.isfinite(metrics["wape_stage_b_cal_guarded"])

    table = summarize([result])
    assert table.loc[0, "window"] == "Aban 1404"
    buckets = lead_bucket_wape(result)
    assert buckets["demand_share"].sum() == pytest.approx(1.0)


def test_backtest_refuses_a_window_the_logs_cannot_resolve(visible, holidays, fast_model):
    with pytest.raises(ValueError, match="cannot be scored"):
        run_backtest(visible, holidays, AZAR, fast_model)


def test_train_save_load_forecast(
    tmp_path, trained, visible_with_target, holidays, synthetic, clusters
):
    forecaster = trained
    forecast = forecaster.forecast(visible_with_target, holidays)

    n_clusters = visible_with_target["cluster_code"].nunique()
    assert list(forecast.columns) == ["cluster_code", "checkin", "predicted_demand"]
    assert len(forecast) == n_clusters * 30
    assert forecast["predicted_demand"].dtype.kind == "i"
    assert (forecast["predicted_demand"] >= 0).all()
    assert forecast["checkin"].min() == AZAR.start.date()
    assert forecast["checkin"].max() == AZAR.end.date()

    # artifacts are plain files and reload to an identical forecaster
    forecaster.save(tmp_path / "artifacts")
    assert {p.suffix for p in (tmp_path / "artifacts").iterdir()} == {".txt", ".json", ".csv"}
    reloaded = TrainedForecaster.load(tmp_path / "artifacts")
    pd.testing.assert_frame_equal(forecast, reloaded.forecast(visible_with_target, holidays))
    assert reloaded.target == AZAR

    # score against the hidden truth at cluster level, versus the naive pooled-curve scaling
    truth = synthetic["target_truth"].merge(clusters, on="city_code")
    truth = truth.groupby(["cluster_code", "checkin"], as_index=False)["demand"].sum()
    truth["checkin"] = truth["checkin"].dt.date
    scored = forecast.merge(truth, on=["cluster_code", "checkin"])
    assert len(scored) == len(forecast)

    lead = (pd.to_datetime(scored["checkin"]) - AZAR.cutoff).dt.days.to_numpy()
    # check-ins nobody searched for before the cutoff are absent from the grid -> zero observed
    cells = scored.merge(
        visible_with_target.assign(checkin=visible_with_target["checkin"].dt.date),
        on=["cluster_code", "checkin"],
        how="left",
    )
    cells[LEAD_COLUMNS] = cells[LEAD_COLUMNS].fillna(0)
    partial = cells[LEAD_COLUMNS].to_numpy()[np.arange(len(cells)), lead]
    naive = partial / forecaster.components.reveal.naive_fraction(lead)
    assert wape(scored["demand"], scored["predicted_demand"]) < wape(scored["demand"], naive)


def test_guardrail_changes_only_high_volume_cells(trained, visible_with_target, holidays):
    plain = trained.forecast(visible_with_target, holidays)
    guarded = trained.forecast(visible_with_target, holidays, guardrail=True)
    assert plain[["cluster_code", "checkin"]].equals(guarded[["cluster_code", "checkin"]])

    pairs = (
        visible_with_target[["cluster_code"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"checkin": pd.date_range(AZAR.start, AZAR.end)}), how="cross")
    )
    cells = build_inference_frame(
        pairs, AZAR.cutoff, visible_with_target, trained.components, holidays
    )
    cells["checkin"] = cells["checkin"].dt.date
    stage_a = (
        cells.set_index(["cluster_code", "checkin"])["stageA_estimate"]
        .reindex(pd.MultiIndex.from_frame(plain[["cluster_code", "checkin"]]))
        .to_numpy()
    )
    small = stage_a < GUARD_MIN_SCALE
    assert (plain["predicted_demand"][small] == guarded["predicted_demand"][small]).all()
    large = ~small
    assert (guarded["predicted_demand"][large] >= np.floor(GUARD_LOW * stage_a[large])).all()
    assert (guarded["predicted_demand"][large] <= np.ceil(GUARD_HIGH * stage_a[large])).all()


def test_forecaster_rejects_artifacts_with_a_different_feature_list(
    tmp_path, visible, holidays, fast_model
):
    forecaster = train_forecaster(visible, holidays, ABAN, fast_model)
    forecaster.save(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["features"] = metadata["features"][:-1]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature list differs"):
        TrainedForecaster.load(tmp_path)


def test_cli_full_workflow(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    main(
        ["synth", "--out", str(data), "--n-cities", "12", "--scale", "0.4", "--start", "2024-09-01"]
    )
    clusters = out / "clusters.csv"
    main(["cluster", "--data-dir", str(data), "--out", str(clusters), "--floor", "100"])
    common = ["--data-dir", str(data), "--clusters", str(clusters)]

    main(["backtest", *common, "--windows", "1404-08", "--out-dir", str(out / "bt")])
    summary = json.loads((out / "bt" / "summary.json").read_text())
    assert summary[0]["window"] == "Aban 1404"
    assert (out / "bt" / "backtest_1404-08.csv").exists()

    main(["train", *common, "--artifacts", str(out / "artifacts")])
    main(
        [
            "forecast",
            *common,
            "--artifacts",
            str(out / "artifacts"),
            "--out",
            str(out / "results.csv"),
        ]
    )
    results = pd.read_csv(out / "results.csv")
    assert list(results.columns) == ["cluster_code", "checkin", "predicted_demand"]
    assert results["checkin"].nunique() == 30
    assert results["cluster_code"].nunique() == pd.read_csv(clusters)["cluster_code"].nunique()


def test_cli_reports_missing_data_helpfully(tmp_path):
    with pytest.raises(SystemExit, match=r"search_data\.csv"):
        main(["backtest", "--data-dir", str(tmp_path), "--clusters", str(tmp_path / "c.csv")])
