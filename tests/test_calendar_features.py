import numpy as np
import pandas as pd

from demand_forecasting.calendar_features import (
    HOLIDAY_FEATURES_PATH,
    add_calendar_features,
    build_holiday_features,
    load_base_calendar,
)


def _toy_calendar(off: list[int]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(off))
    return pd.DataFrame({"gregorian_date": dates, "is_day_off": np.array(off, dtype=bool)})


def test_block_features_on_a_toy_calendar():
    #            idx: 0  1  2  3  4  5  6  7
    cal = _toy_calendar([0, 0, 1, 1, 0, 1, 0, 0])
    out = build_holiday_features(cal)

    assert out["in_holiday_block"].tolist() == [False, False, True, True, False, True, False, False]
    # idx 4 is a single working day squeezed between two blocks -> bridge day
    assert out["is_bridge_day"].tolist() == [False] * 4 + [True] + [False] * 3
    # block lengths: [2,3] has length 2, [5] has length 1
    assert out["current_block_length"].fillna(0).tolist() == [0, 0, 2, 2, 0, 1, 0, 0]
    # idx 1: next block starts in 1 day; nothing precedes it
    assert out.loc[1, "days_until_next_holiday_start"] == 1
    assert np.isnan(out.loc[1, "days_since_last_holiday_end"])
    # idx 6: previous block ended 1 day ago, no next block
    assert out.loc[6, "days_since_last_holiday_end"] == 1
    assert out.loc[6, "days_to_nearest_holiday"] == 1
    # ties go to the previous block: idx 4 is 1 day from both (lengths 2 and 1)
    assert out.loc[4, "nearest_block_length"] == 2


def test_rebuilt_features_match_the_shipped_table_exactly():
    """The bundled holiday_interval_features.csv is reproducible from the base calendar."""
    shipped = pd.read_csv(HOLIDAY_FEATURES_PATH, parse_dates=["gregorian_date"])
    rebuilt = build_holiday_features(load_base_calendar())
    assert list(rebuilt.columns) == list(shipped.columns)
    for column in shipped.columns:
        a, b = rebuilt[column], shipped[column]
        if column in {"jalali_date", "day_of_week", "events"}:
            assert (a.fillna("") == b.fillna("")).all(), column
        elif a.dtype == bool or b.dtype == bool:
            assert (a.astype(bool) == b.astype(bool)).all(), column
        elif column == "gregorian_date":
            assert (a == b).all()
        else:
            assert np.array_equal(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True
            ), column


def test_iran_weekday_order_and_weekend(holidays):
    # 2025-11-22 is a Saturday (start of the Iranian week), 27th a Thursday, 28th a Friday
    frame = pd.DataFrame({"checkin": pd.to_datetime(["2025-11-22", "2025-11-27", "2025-11-28"])})
    out = add_calendar_features(frame, "checkin", holidays)
    assert out["dow"].tolist() == [0, 5, 6]
    assert out["is_weekend"].tolist() == [0, 1, 1]
    assert out["jalali_month"].tolist() == [9, 9, 9]
    assert out["month"].tolist() == [11, 11, 11]


def test_nowruz_1402_is_one_13_day_block(holidays):
    # 1402-01-01 .. 1402-01-13 (2023-03-21 .. 2023-04-02) is a single run of days off
    frame = pd.DataFrame({"checkin": pd.to_datetime(["2023-03-21", "2023-04-02", "2023-04-03"])})
    out = add_calendar_features(frame, "checkin", holidays)
    assert out["is_holiday"].tolist() == [1, 1, 0]
    assert out["in_holiday_block"].tolist() == [1, 1, 0]
    assert out["nearest_block_length"].tolist() == [13, 13, 13]  # the day after is 1 day from it
    assert out["days_to_nearest_holiday"].tolist() == [0, 0, 1]


def test_dates_outside_the_calendar_get_neutral_values(holidays):
    frame = pd.DataFrame({"checkin": pd.to_datetime(["2035-01-01"])})
    out = add_calendar_features(frame, "checkin", holidays)
    assert out.loc[0, ["is_holiday", "is_day_off", "in_holiday_block", "is_bridge_day"]].sum() == 0
    assert out.loc[0, "days_to_nearest_holiday"] == 30
    assert out.loc[0, "nearest_block_length"] == 1


def test_add_calendar_features_preserves_row_order_and_count(holidays):
    dates = pd.to_datetime(["2025-05-01", "2025-01-01", "2025-05-01", "2025-03-21"])
    out = add_calendar_features(
        pd.DataFrame({"checkin": dates, "x": [1, 2, 3, 4]}), "checkin", holidays
    )
    assert out["x"].tolist() == [1, 2, 3, 4]
    assert (out["checkin"] == dates).all()
