"""Synthetic search logs with the same schema and the same difficulty as the real data.

The original data are private and cannot be redistributed, so this module generates a stand-in
that lets anyone run, test and demo the full pipeline. It reproduces the properties the model
has to cope with:

* **heavy-tailed city sizes** - a few huge cities and many near-empty ones (so clustering and
  the volume floor matter);
* **weekday, holiday, seasonal and Nowruz effects** driven by the real Iranian calendar;
* **resort cities** with a sharp summer peak;
* **right-censored demand** - each search is logged ``lead`` days before the stay, with longer
  leads on Thursday/Friday and around holidays, so a future date is only partly observed;
* a **target month** whose logs stop at the cutoff (``evaluation.csv``) while the eventual
  true demand is written to ``target_truth.csv`` so forecasts can be scored.

It makes no attempt to match the real data's absolute volumes or exact dynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_features import HOLIDAY_FEATURES_PATH, add_calendar_features, load_holiday_features
from .config import MAX_LEAD

logger = logging.getLogger(__name__)

# Demand level by Iranian weekday (Saturday .. Friday): Wednesday/Thursday peak.
_DOW_LEVEL = np.array([0.85, 0.85, 0.90, 0.95, 1.15, 1.40, 1.00])
# Typical booking horizon (days) by weekday: weekend stays are planned further ahead.
_DOW_LEAD_SCALE = np.array([6.0, 6.0, 6.0, 6.5, 7.5, 10.0, 12.0])


@dataclass(frozen=True)
class SyntheticConfig:
    n_cities: int = 60
    n_provinces: int = 5
    seed: int = 7
    start: str = "2023-11-01"
    """First check-in date (the real logs start on 2023-11-01)."""
    cutoff: str = "2025-11-21"
    """Last day with logs = end of Aban 1404, the forecast origin."""
    target_days: int = 30
    """Length of the target month (Azar 1404 has 30 days)."""
    scale: float = 1.0
    """Multiplies every city's demand level (lower = sparser data and smaller files)."""


def _season(jalali_month: np.ndarray, resort: np.ndarray) -> np.ndarray:
    """(n_cities, n_days) multiplicative seasonality; resort cities peak in Mordad/Shahrivar."""
    month = jalali_month[None, :].astype(float)
    ordinary = 1.0 + 0.25 * np.sin(2 * np.pi * (month - 3.0) / 12.0)
    resort_peak = 1.0 + 1.6 * np.exp(-(((month - 5.5) / 1.1) ** 2))
    return np.where(resort[:, None], resort_peak, ordinary)


def generate(
    config: SyntheticConfig | None = None, holidays: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Generate ``search_data``, ``evaluation``, ``cities`` and ``target_truth`` frames."""
    config = config or SyntheticConfig()
    holidays = load_holiday_features(HOLIDAY_FEATURES_PATH) if holidays is None else holidays
    rng = np.random.default_rng(config.seed)
    cutoff = pd.Timestamp(config.cutoff)
    checkins = pd.date_range(config.start, cutoff + pd.Timedelta(days=config.target_days))
    n_days, n_cities = len(checkins), config.n_cities

    # ---- cities: provinces are blobs in lat/long space
    centres = rng.uniform([28.0, 46.0], [38.0, 60.0], size=(config.n_provinces, 2))
    province_idx = rng.integers(0, config.n_provinces, size=n_cities)
    coords = centres[province_idx] + rng.normal(0.0, 0.5, size=(n_cities, 2))
    city_code = rng.choice(np.arange(1000, 9999), size=n_cities, replace=False)
    cities = pd.DataFrame(
        {
            "city_code": city_code,
            "province_code": 100 + 37 * province_idx,
            "lat": coords[:, 0].round(6),
            "long": coords[:, 1].round(6),
        }
    )
    base = np.exp(rng.normal(1.6, 1.8, size=n_cities)) * config.scale  # searches/day
    resort = rng.random(n_cities) < 0.2

    # ---- expected demand per (city, check-in date)
    cal = add_calendar_features(pd.DataFrame({"checkin": checkins}), "checkin", holidays)
    dow = cal["dow"].to_numpy()
    is_holiday = cal["is_holiday"].to_numpy()
    in_block = cal["in_holiday_block"].to_numpy()
    nowruz = ((cal["jalali_month"] == 1) & (cal["week_of_month"] <= 2)).to_numpy()
    calendar_level = (
        _DOW_LEVEL[dow] * (1 + 0.35 * is_holiday + 0.15 * in_block) * np.where(nowruz, 2.2, 1.0)
    )
    years = (checkins - checkins[0]).days.to_numpy() / 365.25
    growth = 1.0 + 0.25 * years
    mu = base[:, None] * _season(cal["jalali_month"].to_numpy(), resort)
    mu = mu * calendar_level[None, :] * growth[None, :]
    demand = rng.poisson(mu * rng.gamma(shape=6.0, scale=1 / 6.0, size=mu.shape))

    # ---- spread each date's demand over lead times (exponential booking horizon)
    horizon = _DOW_LEAD_SCALE[dow] + 4.0 * is_holiday + np.where(nowruz, 8.0, 0.0)
    leads = np.arange(MAX_LEAD + 1)
    lead_prob = np.exp(-leads[None, :] / horizon[:, None])
    lead_prob /= lead_prob.sum(axis=1, keepdims=True)

    log_dates, city_idx, check_idx, counts = [], [], [], []
    for d in range(n_days):
        per_lead = rng.multinomial(demand[:, d], lead_prob[d])  # (n_cities, MAX_LEAD + 1)
        city_i, lead_i = np.nonzero(per_lead)
        log_dates.append(checkins[d].to_datetime64() - lead_i.astype("timedelta64[D]"))
        city_idx.append(city_i)
        check_idx.append(np.full(len(city_i), d))
        counts.append(per_lead[city_i, lead_i])

    logs = pd.DataFrame(
        {
            "log_date": np.concatenate(log_dates),
            "city_code": city_code[np.concatenate(city_idx)],
            "checkin": checkins.to_numpy()[np.concatenate(check_idx)],
            "search_count": np.concatenate(counts),
        }
    )
    history = logs["checkin"] <= cutoff
    partial = (logs["checkin"] > cutoff) & (logs["log_date"] <= cutoff)
    target_mask = checkins > cutoff
    truth = pd.DataFrame(
        {
            "city_code": np.repeat(city_code, target_mask.sum()),
            "checkin": np.tile(checkins[target_mask], n_cities),
            "demand": demand[:, target_mask].reshape(-1),
        }
    )
    logger.info(
        "synthetic: %d cities, %d check-in dates, %s history rows, %s partial target rows",
        n_cities,
        n_days,
        f"{history.sum():,}",
        f"{partial.sum():,}",
    )
    return {
        "search_data": logs[history].reset_index(drop=True),
        "evaluation": logs[partial].reset_index(drop=True),
        "cities": cities,
        "target_truth": truth,
    }


def write_dataset(out_dir: str | Path, config: SyntheticConfig | None = None) -> dict[str, Path]:
    """Generate a dataset and write it as the CSV files the pipeline expects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in generate(config).items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
