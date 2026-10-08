"""Helpers for the Persian (Jalali / Solar Hijri) calendar.

Iranian demand is driven by the Persian calendar (Nowruz, Yalda, the Shahrivar/Mehr travel
season ...), so every calendar feature is derived from Jalali dates. Gregorian <-> Jalali
conversion is delegated to :mod:`convertdate` and cached per distinct date, because the
training panel repeats the same few hundred dates millions of times.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from convertdate import persian

from .config import JALALI_MONTH_NAMES

# Days elapsed before the first day of each Jalali month (months 1-6 have 31 days,
# months 7-11 have 30). Esfand's own length never enters a day-of-year sum.
_DAYS_BEFORE_MONTH = np.array([0, 31, 62, 93, 124, 155, 186, 216, 246, 276, 306, 336])


@cache
def _from_gregorian(year: int, month: int, day: int) -> tuple[int, int, int]:
    return persian.from_gregorian(year, month, day)


def jalali_parts(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Jalali calendar parts for every date, aligned positionally with ``dates``.

    Returns a frame with ``jalali_year``, ``jalali_month``, ``jalali_day``, ``jalali_week``
    (week of the Jalali year, 1-53) and ``week_of_month`` (1-5).
    """
    idx = pd.DatetimeIndex(dates)
    unique = idx.unique()
    table = pd.DataFrame(
        [_from_gregorian(d.year, d.month, d.day) for d in unique],
        index=unique,
        columns=["jalali_year", "jalali_month", "jalali_day"],
    )
    parts = table.reindex(idx)
    month = parts["jalali_month"].to_numpy()
    day = parts["jalali_day"].to_numpy()
    day_of_year = _DAYS_BEFORE_MONTH[month - 1] + day
    parts["jalali_week"] = (day_of_year - 1) // 7 + 1
    parts["week_of_month"] = (day - 1) // 7 + 1
    return parts.reset_index(drop=True)


def jalali_to_timestamp(year: int, month: int, day: int) -> pd.Timestamp:
    """Gregorian midnight timestamp of a Jalali date."""
    return pd.Timestamp(*persian.to_gregorian(year, month, day))


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day (inclusive, Gregorian) of a Jalali month; leap years handled."""
    first = jalali_to_timestamp(year, month, 1)
    last = jalali_to_timestamp(year, month, persian.month_length(year, month))
    return first, last


def previous_month_end(year: int, month: int) -> pd.Timestamp:
    """Last day of the Jalali month before ``(year, month)``.

    This is the forecast origin ("as of Aban 30th") when predicting the whole of a month.
    """
    prev_year, prev_month = (year - 1, 12) if month == 1 else (year, month - 1)
    return month_bounds(prev_year, prev_month)[1]


@dataclass(frozen=True)
class JalaliMonth:
    """A Jalali (year, month) - the unit in which forecast windows are specified."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"Jalali month must be in 1..12, got {self.month}")

    @classmethod
    def parse(cls, text: str) -> JalaliMonth:
        """Parse ``"1404-09"`` (or ``"1404/9"``) into a :class:`JalaliMonth`."""
        try:
            year, month = text.replace("/", "-").split("-")
            return cls(int(year), int(month))
        except ValueError as exc:
            raise ValueError(f"Expected a Jalali month like '1404-09', got {text!r}") from exc

    @property
    def label(self) -> str:
        return f"{JALALI_MONTH_NAMES[self.month - 1]} {self.year}"

    @property
    def start(self) -> pd.Timestamp:
        return month_bounds(self.year, self.month)[0]

    @property
    def end(self) -> pd.Timestamp:
        return month_bounds(self.year, self.month)[1]

    @property
    def cutoff(self) -> pd.Timestamp:
        """Forecast origin: the last day of the preceding month."""
        return previous_month_end(self.year, self.month)

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}"
