"""Shared fixtures. A small synthetic dataset is generated once per test session."""

from __future__ import annotations

import pandas as pd
import pytest

from demand_forecasting.calendar_features import load_holiday_features
from demand_forecasting.clustering import ClusteringConfig, cluster_cities
from demand_forecasting.config import ModelConfig
from demand_forecasting.data import aggregate_to_clusters, build_visible_demand
from demand_forecasting.synthetic import SyntheticConfig, generate

# Quick-to-train settings so the end-to-end tests take seconds, not minutes.
FAST_MODEL = ModelConfig(num_boost_round=120, early_stopping_rounds=20)


@pytest.fixture(scope="session")
def holidays() -> pd.DataFrame:
    return load_holiday_features()


@pytest.fixture(scope="session")
def synthetic(holidays) -> dict[str, pd.DataFrame]:
    config = SyntheticConfig(n_cities=14, n_provinces=3, seed=3, start="2024-09-01", scale=0.5)
    return generate(config, holidays)


@pytest.fixture(scope="session")
def clusters(synthetic) -> pd.DataFrame:
    return cluster_cities(
        synthetic["cities"], synthetic["search_data"], ClusteringConfig(floor=100)
    )


@pytest.fixture(scope="session")
def visible(synthetic, clusters) -> pd.DataFrame:
    """History only (no partial target-month rows): what a backtest sees."""
    return build_visible_demand(aggregate_to_clusters(synthetic["search_data"], clusters), clusters)


@pytest.fixture(scope="session")
def visible_with_target(synthetic, clusters) -> pd.DataFrame:
    """History plus the partially observed target month: what production forecasting sees."""
    logs = pd.concat([synthetic["search_data"], synthetic["evaluation"]], ignore_index=True)
    return build_visible_demand(aggregate_to_clusters(logs, clusters), clusters)


@pytest.fixture(scope="session")
def fast_model() -> ModelConfig:
    return FAST_MODEL
