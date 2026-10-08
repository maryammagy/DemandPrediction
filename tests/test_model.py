import numpy as np
import pandas as pd
import pytest

from demand_forecasting.features import FEATURES, FittedComponents, build_training_panel
from demand_forecasting.jalali import JalaliMonth
from demand_forecasting.model import (
    Calibration,
    StageBModel,
    apply_guardrail,
    bucket_index,
    fit_stage_b,
)

WINDOW = JalaliMonth(1404, 8)


def test_bucket_index_uses_right_closed_intervals():
    leads = np.array([0, 1, 3, 4, 7, 8, 14, 15, 21, 22, 30, 31])
    assert bucket_index(leads).tolist() == [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1]


def test_calibration_scales_up_underprediction_per_bucket():
    leads = np.array([2, 2, 10, 10])
    truth = np.array([110.0, 110.0, 100.0, 100.0])
    pred = np.array([100.0, 100.0, 50.0, 50.0])
    cal = Calibration.fit(truth, pred, leads)
    assert cal.by_bucket["(0, 3]"] == pytest.approx(1.1)
    assert cal.by_bucket["(7, 14]"] == pytest.approx(2.0)
    assert cal.global_factor == pytest.approx(420 / 300)
    np.testing.assert_allclose(cal.factors(np.array([3, 14])), [1.1, 2.0])


def test_calibration_is_floored_at_one_never_scales_down():
    leads = np.array([2, 10])
    cal = Calibration.fit(
        np.array([50.0, 50.0]), np.array([100.0, 100.0]), leads
    )  # over-prediction
    assert cal.global_factor == 1.0
    assert set(cal.by_bucket.values()) == {1.0}


def test_calibration_falls_back_to_global_for_unseen_or_out_of_range_leads():
    cal = Calibration.fit(np.array([120.0]), np.array([100.0]), np.array([2]))
    assert cal.factors(np.array([20, 0, 45])).tolist() == [cal.global_factor] * 3


def test_calibration_round_trips_through_json_format():
    cal = Calibration.fit(np.array([120.0, 90.0]), np.array([100.0, 100.0]), np.array([2, 10]))
    payload = cal.to_dict()
    assert list(payload["cal_by_lead_bucket"]) == ["(0, 3]", "(7, 14]"]  # pandas interval labels
    assert Calibration.from_dict(payload) == cal


def test_guardrail_only_clips_high_volume_cells():
    stage_a = np.array([2000.0, 2000.0, 2000.0, 100.0])
    stage_b = np.array([500.0, 2100.0, 9000.0, 9000.0])
    out = apply_guardrail(stage_b, stage_a)
    np.testing.assert_allclose(
        out, [1400.0, 2100.0, 3000.0, 9000.0]
    )  # 0.7x / as-is / 1.5x / small cell


def test_stage_b_trains_predicts_and_calibrates(visible, holidays, fast_model):
    components = FittedComponents.fit(visible, WINDOW.start)
    panel = build_training_panel(visible, components, holidays, fast_model.leads)
    model, report = fit_stage_b(panel, WINDOW.start, fast_model)

    assert isinstance(model, StageBModel)
    assert model.features == FEATURES
    assert report.best_iteration > 0
    assert report.n_train > report.n_validation > 0
    assert np.isfinite(report.wape_stage_a) and np.isfinite(report.wape_stage_b)
    assert (model.calibration.factors(np.arange(1, 31)) >= 1.0).all()

    held_out = panel.tail(500)
    raw, calibrated = model.predict(held_out, calibrated=False), model.predict(held_out)
    assert (raw >= 0).all()
    assert (calibrated >= raw - 1e-12).all()  # factors are >= 1


def test_stage_b_training_is_deterministic(visible, holidays, fast_model):
    components = FittedComponents.fit(visible, WINDOW.start)
    panel = build_training_panel(visible, components, holidays, fast_model.leads)
    a, _ = fit_stage_b(panel, WINDOW.start, fast_model)
    b, _ = fit_stage_b(panel, WINDOW.start, fast_model)
    sample = panel.sample(300, random_state=1)
    assert np.array_equal(a.predict_raw(sample), b.predict_raw(sample))


def test_fit_stage_b_needs_both_folds(visible, holidays, fast_model):
    components = FittedComponents.fit(visible, WINDOW.start)
    panel = build_training_panel(visible, components, holidays, fast_model.leads)
    with pytest.raises(ValueError, match="not enough history"):
        fit_stage_b(panel, pd.Timestamp("2030-01-01") + pd.Timedelta(days=10_000), fast_model)
