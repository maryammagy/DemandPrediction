import pandas as pd
import pytest

from demand_forecasting.jalali import JalaliMonth, jalali_parts, month_bounds, previous_month_end


def test_known_conversions():
    parts = jalali_parts(pd.to_datetime(["2025-03-21", "2025-11-21", "2025-11-22"]))
    assert parts[["jalali_year", "jalali_month", "jalali_day"]].values.tolist() == [
        [1404, 1, 1],  # Nowruz 1404
        [1404, 8, 30],  # Aban 30, the forecast origin
        [1404, 9, 1],  # Azar 1
    ]


def test_week_of_year_and_month_boundaries():
    dates = pd.to_datetime(["2025-03-21", "2025-03-27", "2025-03-28", "2025-04-20"])
    parts = jalali_parts(dates)
    assert parts["jalali_week"].tolist() == [1, 1, 2, 5]  # 1404-01-01, -07, -08, -31
    assert parts["week_of_month"].tolist() == [1, 1, 2, 5]


def test_repeated_dates_are_converted_consistently():
    dates = pd.to_datetime(["2025-11-22"] * 3 + ["2025-11-23"])
    assert jalali_parts(dates)["jalali_day"].tolist() == [1, 1, 1, 2]


def test_azar_1404_window():
    azar = JalaliMonth.parse("1404-09")
    assert azar.label == "Azar 1404"
    assert str(azar.start.date()) == "2025-11-22"
    assert str(azar.end.date()) == "2025-12-21"
    assert str(azar.cutoff.date()) == "2025-11-21"  # last day of Aban
    assert (azar.end - azar.start).days + 1 == 30


def test_leap_year_esfand_has_30_days():
    # 1403 is a leap year: its Esfand ends on 2025-03-20, the day before Nowruz 1404.
    assert str(month_bounds(1403, 12)[1].date()) == "2025-03-20"
    assert str(previous_month_end(1404, 1).date()) == "2025-03-20"


def test_parse_accepts_slash_and_rejects_garbage():
    assert JalaliMonth.parse("1404/9") == JalaliMonth(1404, 9)
    with pytest.raises(ValueError):
        JalaliMonth.parse("2025-11-22")
    with pytest.raises(ValueError):
        JalaliMonth(1404, 13)
