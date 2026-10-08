# Data

> **The original dataset is confidential and is not part of this repository.**
> It was provided for a hackathon-style assignment under terms that prohibit redistribution, so
> neither the raw files nor anything that embeds them (trained models, cluster statistics,
> forecasts) is committed - `.gitignore` blocks `data/raw/`, `artifacts/` and `outputs/`.
>
> You can still run the entire pipeline: [`demand-forecast synth`](#option-a-synthetic-data-no-access-needed)
> generates a stand-in dataset with the same schema and the same modelling difficulty.

## What the pipeline expects

Put these files in `data/raw/` (or any folder you pass with `--data-dir`).

### `search_data.csv` - aggregated search logs

One row per **(log date, city, check-in date)** that had at least one search. A missing row means
*zero searches*.

| column | type | description |
|---|---|---|
| `log_date` | date | day the searches happened |
| `city_code` | int | anonymised city identifier |
| `checkin` | date | intended check-in date of the searched stay |
| `search_count` | int (> 0) | number of searches |

* `lead = checkin - log_date` must lie in **0 ... 59 days** (rows outside are ignored).
* Every check-in in this file must be *fully resolved*: all of its searches are already logged
  (i.e. `checkin <=` the last log date). This is what makes backtesting possible.

### `cities.csv` - static city metadata

| column | type | description |
|---|---|---|
| `city_code` | int | same identifier as above |
| `province_code` | int | province; sparse cities are only merged **within** a province |
| `lat`, `long` | float | coordinates; used to find the geographically nearest cluster |

### `evaluation.csv` - partial logs of the target month *(forecasting only)*

Same schema as `search_data.csv`, but for check-ins in the month to be forecast and only for
searches logged **up to the forecast origin**. It is concatenated with `search_data.csv` by
`demand-forecast forecast`, and never used for training or backtesting.

### `clusters.csv` - city → cluster mapping *(produced by `demand-forecast cluster`)*

| column | description |
|---|---|
| `city_code` | city |
| `cluster_code` | the code of one of the cluster's member cities |

### Profile of the original data (for orientation only)

| | |
|---|---|
| search-log rows | ~3.3 M |
| cities / provinces | 321 / 7 |
| check-in range | 2023-11-01 … 2025-11-21 (752 days) |
| lead times observed | 0 … 59 days |
| forecast target | the 30 check-in days of Azar 1404 (2025-11-22 … 2025-12-21), as of Aban 30th |

## Bundled calendar data (public)

`src/demand_forecasting/resources/` ships two small files that **are** safe to share - they encode
Iranian public holidays, not task data:

| file | content |
|---|---|
| `iran_holidays_1402_1404.csv` | one row per day, 21 Mar 2023 → 20 Mar 2026: `is_holiday`, `is_weekend` (Thursday + Friday), `is_day_off`, Persian/English event labels |
| `holiday_interval_features.csv` | the same days with derived *block* features (length of the surrounding run of days off, distance to the nearest one, bridge days, ...) |

**Provenance.** The holidays are based on the official annual national calendars (University of
Tehran Calendar Center). Islamic (lunar) holidays use the dates printed in those calendars, which
can shift by a day if the moon-sighting authorities revise them. The table deliberately deviates
from the strictly official list on **8 days**, as *modelling conventions*:

* **5-11 Farvardin 1402** are flagged so that Nowruz 1402 forms one continuous 13-day block
  (the real travel window), instead of the official 1-4 and 12-13 Farvardin - see experiment #1
  in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md);
* **30 Esfand 1403** (last day of the year) is flagged as a holiday.

Every strictly official holiday date in 1402-1404 is present in the table.

The second file is **fully reproducible** from the first - a test asserts they match exactly:

```bash
demand-forecast calendar --out outputs/holiday_interval_features.csv
```

To forecast beyond Esfand 1404, extend `iran_holidays_1402_1404.csv` with the new year's days
(same columns) and rebuild. Dates outside the calendar get neutral holiday features instead of
failing.

## Option A: synthetic data (no access needed)

```bash
demand-forecast synth --out data/synthetic            # 60 cities by default (the README's 40-city quick start runs everything in ~45 s)
demand-forecast cluster  --data-dir data/synthetic --floor 200
```

The generator ([`synthetic.py`](../src/demand_forecasting/synthetic.py)) reproduces the properties
the model has to cope with: heavy-tailed city sizes, weekday / holiday / seasonal / Nowruz effects
driven by the real calendar, resort cities with a sharp summer peak, longer booking horizons for
weekend stays, and a target month that is only partially observed at the forecast origin. It also
writes `target_truth.csv` - the eventual demand of the target month - so forecasts can be scored.
It does **not** try to match the real data's absolute volumes, so its WAPE values are *not*
comparable with the results reported in the main README.

## Option B: your own data

Any search/booking log that can be aggregated to the schema above works. Things to check:

1. **Reference year for clustering.** `demand-forecast cluster` uses the Jalali year of the latest
   check-in (override with `--year`) and requires a minimum monthly demand (`--floor`).
2. **Target month.** `--target YYYY-MM` is a *Jalali* month; the forecast origin is the last day of
   the previous month. For another calendar system, replace `jalali.py` and the holiday table.
3. **Volume scale.** Two constants assume the original magnitude: the guardrail's `GUARD_MIN_SCALE`
   (1,000 searches) and the clustering floor (500 / month). Rescale them for much smaller or
   larger platforms.
