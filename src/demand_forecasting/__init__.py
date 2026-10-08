"""Forecast future demand from partially observed search logs.

The package implements a two-stage model:

* **Stage A** - a *reveal curve* that grosses up the searches observed so far for a
  future check-in date into an estimate of its eventual total demand.
* **Stage B** - a LightGBM regressor that corrects Stage A using calendar, holiday and
  cluster-history features, followed by a leakage-free multiplicative calibration.

See ``docs/METHODOLOGY.md`` for the full write-up.
"""

__version__ = "1.0.0"
