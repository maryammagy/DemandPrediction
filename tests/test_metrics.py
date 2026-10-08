import math

import numpy as np

from demand_forecasting.metrics import wape


def test_perfect_forecast_has_zero_error():
    y = np.array([10.0, 0.0, 5.0])
    assert wape(y, y) == 0.0


def test_known_value():
    # |12-10| + |3-5| + |0-0| = 4, divided by 15
    assert math.isclose(wape([10, 5, 0], [12, 3, 0]), 4 / 15)


def test_is_volume_weighted_not_a_per_cell_average():
    # The tiny cell is 100% wrong, the big one only 1% wrong: WAPE stays close to 1%.
    assert wape([1, 1000], [2, 990]) < 0.02


def test_all_zero_truth_is_nan():
    assert math.isnan(wape([0, 0], [1, 2]))


def test_accepts_pandas_series():
    import pandas as pd

    assert wape(pd.Series([1.0, 2.0]), pd.Series([1.0, 4.0])) == 2 / 3
