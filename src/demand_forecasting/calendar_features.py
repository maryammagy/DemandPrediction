"""Calendar and holiday features.

Two layers:

1. :func:`build_holiday_features` turns a base day-by-day calendar (``is_holiday``,
   ``is_weekend``, ``is_day_off``) into *block* features - how long the surrounding run of
   days off is, how far the nearest one is, whether a day is a "bridge" between two of them.
   The result ships as ``data/external/holiday_interval_features.csv``.
2. :func:`add_calendar_features` attaches those columns plus Gregorian/Jalali date parts to
   any frame with a check-in date column.

Iran's weekend is **Thursday + Friday** and its week runs **Saturday (0) to Friday (6)**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .jalali import jalali_parts

HOLIDAY_FEATURE_COLUMNS = [
    "is_holiday",
    "is_weekend",
    "is_day_off",
    "is_bridge_day",
    "in_holiday_block",
    "days_to_nearest_holiday",
    "nearest_block_length",
]
"""Columns of the holiday table consumed by the model."""

_BOOLEAN_FEATURES = ["is_holiday", "is_weekend", "is_day_off", "is_bridge_day", "in_holiday_block"]
_NO_NEIGHBOUR_DISTANCE = 30  # "no holiday block known nearby" fill value (days)
_NO_NEIGHBOUR_LENGTH = 1

RESOURCE_DIR = Path(__file__).parent / "resources"
BASE_CALENDAR_PATH = RESOURCE_DIR / "iran_holidays_1402_1404.csv"
HOLIDAY_FEATURES_PATH = RESOURCE_DIR / "holiday_interval_features.csv"


def load_holiday_features(path: str | Path = HOLIDAY_FEATURES_PATH) -> pd.DataFrame:
    """Read the holiday-feature table (``gregorian_date`` + model columns).

    Defaults to the table bundled with the package (Jalali years 1402-1404).
    """
    return pd.read_csv(
        path,
        parse_dates=["gregorian_date"],
        usecols=["gregorian_date", *HOLIDAY_FEATURE_COLUMNS],
    )


def load_base_calendar(path: str | Path = BASE_CALENDAR_PATH) -> pd.DataFrame:
    """Read a base day-by-day calendar (the input of :func:`build_holiday_features`)."""
    return pd.read_csv(path, parse_dates=["gregorian_date"])


def build_holiday_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Derive holiday-block features from a base calendar.

    Parameters
    ----------
    calendar:
        One row per consecutive day with at least a boolean ``is_day_off`` column
        (official holiday *or* weekend). Extra columns (dates, event names ...) are kept.

    Notes
    -----
    A *block* is a maximal run of consecutive days off. All distances are in days:

    * ``days_since_last_holiday_end`` - days since the most recent block's last day
      (0 *on* that last day; NaN before the first block ends).
    * ``days_until_next_holiday_start`` - days until the next block's first day
      (0 *on* that first day).
    * ``nearest_block_length`` / ``days_to_nearest_holiday`` - length of / distance to the
      closest block (ties go to the previous one; 0 inside a block).
    * ``is_bridge_day`` - a working day squeezed between two blocks (exactly one day away
      from each), which Iranians often take off as well.

    The derived columns are bit-identical to the ``holiday_interval_features.csv`` that the
    original model was trained with (see ``tests/test_calendar_features.py``).
    """
    off = calendar["is_day_off"].to_numpy(dtype=bool)
    n = len(off)
    position = np.arange(n)

    starts = off & ~np.r_[False, off[:-1]]
    ends = off & ~np.r_[off[1:], False]
    block_id = np.cumsum(starts) * off
    length = pd.Series(off).groupby(block_id).transform("size").to_numpy() * off
    pos_in_block = pd.Series(off.astype(int)).groupby(block_id).cumsum().to_numpy() * off

    last_end = pd.Series(np.where(ends, position, np.nan)).ffill().to_numpy()
    next_start = pd.Series(np.where(starts, position, np.nan)).bfill().to_numpy()
    days_since = position - last_end
    days_until = next_start - position
    prev_len = pd.Series(np.where(ends, length, np.nan)).ffill().to_numpy()
    next_len = pd.Series(np.where(starts, length, np.nan)).bfill().to_numpy()

    since_or_inf = np.where(np.isnan(days_since), np.inf, days_since)
    until_or_inf = np.where(np.isnan(days_until), np.inf, days_until)
    prefer_previous = since_or_inf <= until_or_inf  # ties -> previous block

    out = calendar.copy()
    out["days_since_last_holiday_end"] = days_since
    out["prev_block_length"] = prev_len
    out["days_until_next_holiday_start"] = days_until
    out["next_block_length"] = next_len
    out["in_holiday_block"] = off
    out["current_block_length"] = np.where(off, length, np.nan)
    out["position_in_block"] = np.where(off, pos_in_block, np.nan)
    out["nearest_block_length"] = np.where(
        off, length, np.where(prefer_previous, prev_len, next_len)
    )
    out["days_to_nearest_holiday"] = np.where(off, 0.0, np.minimum(since_or_inf, until_or_inf))
    out["is_bridge_day"] = (~off) & (days_since == 1) & (days_until == 1)
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str, holidays: pd.DataFrame) -> pd.DataFrame:
    """Attach date parts and holiday features keyed on ``df[date_col]``.

    Adds ``month`` (Gregorian), ``dow`` (Iran order, Saturday = 0), ``jalali_month``,
    ``jalali_week`` (week of Jalali year), ``week_of_month`` and the
    :data:`HOLIDAY_FEATURE_COLUMNS`. Dates outside the holiday table get neutral values
    (no holiday, no neighbouring block).
    """
    out = df.merge(
        holidays[["gregorian_date", *HOLIDAY_FEATURE_COLUMNS]],
        left_on=date_col,
        right_on="gregorian_date",
        how="left",
    ).drop(columns="gregorian_date")

    out["month"] = out[date_col].dt.month
    # pandas: Monday=0..Sunday=6  ->  Iran: Saturday=0..Friday=6
    out["dow"] = (out[date_col].dt.dayofweek + 2) % 7
    parts = jalali_parts(out[date_col])
    out["jalali_month"] = parts["jalali_month"].to_numpy()
    out["jalali_week"] = parts["jalali_week"].to_numpy()
    out["week_of_month"] = parts["week_of_month"].to_numpy()

    for col in _BOOLEAN_FEATURES:
        out[col] = out[col].eq(True).astype(int)
    out["nearest_block_length"] = out["nearest_block_length"].fillna(_NO_NEIGHBOUR_LENGTH)
    out["days_to_nearest_holiday"] = out["days_to_nearest_holiday"].fillna(_NO_NEIGHBOUR_DISTANCE)
    return out
