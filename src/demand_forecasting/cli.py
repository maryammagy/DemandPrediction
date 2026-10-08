"""Command-line interface: ``demand-forecast <command>`` (or ``python -m demand_forecasting``).

Typical workflow::

    demand-forecast synth    --out data/synthetic            # (or put the real CSVs in data/raw)
    demand-forecast cluster  --data-dir data/synthetic --floor 200
    demand-forecast backtest --data-dir data/synthetic --clusters outputs/clusters.csv
    demand-forecast train    --data-dir data/synthetic --clusters outputs/clusters.csv
    demand-forecast forecast --data-dir data/synthetic --clusters outputs/clusters.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import DEFAULT_WINDOWS, run_backtest, summarize
from .calendar_features import (
    BASE_CALENDAR_PATH,
    HOLIDAY_FEATURES_PATH,
    build_holiday_features,
    load_base_calendar,
    load_holiday_features,
)
from .clustering import ClusteringConfig, cluster_cities
from .config import ModelConfig
from .data import (
    aggregate_to_clusters,
    build_visible_demand,
    load_cities,
    load_clusters,
    load_search_data,
)
from .forecaster import TrainedForecaster, train_forecaster
from .jalali import JalaliMonth
from .synthetic import SyntheticConfig, write_dataset

logger = logging.getLogger("demand_forecasting")

DEFAULT_TARGET = "1404-09"  # Azar 1404, forecast as of Aban 30th


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise SystemExit(f"error: {path} not found. {hint}")
    return path


def _prepare(
    data_dir: Path, clusters_path: Path, *, include_evaluation: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load logs + cluster mapping and build the visible-demand grid."""
    hint = "See data/README.md, or generate a stand-in with `demand-forecast synth`."
    _require(data_dir / "search_data.csv", hint)
    _require(clusters_path, "Create it with `demand-forecast cluster`.")
    use_evaluation = include_evaluation and (data_dir / "evaluation.csv").exists()
    if include_evaluation and not use_evaluation:
        logger.warning(
            "no evaluation.csv in %s - forecasting without the target month's logs", data_dir
        )
    search = load_search_data(data_dir, include_evaluation=use_evaluation)
    clusters = load_clusters(clusters_path)
    visible = build_visible_demand(aggregate_to_clusters(search, clusters), clusters)
    logger.info(
        "visible-demand grid: %s (cluster, check-in) rows, %d clusters",
        f"{len(visible):,}",
        clusters["cluster_code"].nunique(),
    )
    return visible, clusters


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(sample_weighting=args.sample_weighting)


def cmd_synth(args: argparse.Namespace) -> None:
    config = SyntheticConfig(
        n_cities=args.n_cities, seed=args.seed, scale=args.scale, start=args.start
    )
    for name, path in write_dataset(args.out, config).items():
        print(f"wrote {name:12s} -> {path}")


def cmd_cluster(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    search = load_search_data(data_dir)
    cities = load_cities(data_dir)
    config = ClusteringConfig(
        floor=args.floor,
        jalali_year=args.year,
        fix_zero_volume_centroid=args.fix_zero_volume_centroid,
    )
    clusters = cluster_cities(cities, search, config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(args.out, index=False)
    n_clusters = clusters["cluster_code"].nunique()
    print(f"wrote {n_clusters} clusters for {len(clusters)} cities -> {args.out}")


def cmd_calendar(args: argparse.Namespace) -> None:
    table = build_holiday_features(load_base_calendar(args.calendar))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} days of holiday features -> {args.out}")


def cmd_backtest(args: argparse.Namespace) -> None:
    visible, _ = _prepare(Path(args.data_dir), Path(args.clusters), include_evaluation=False)
    holidays = load_holiday_features(args.holidays)
    windows = [JalaliMonth.parse(w) for w in args.windows] if args.windows else DEFAULT_WINDOWS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    config = _model_config(args)
    for window in windows:
        result = run_backtest(visible, holidays, window, config)
        result.frame.assign(checkin=result.frame["checkin"].dt.date).to_csv(
            out_dir / f"backtest_{window}.csv", index=False
        )
        results.append(result)

    summary = summarize(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict("records"), indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nWAPE by backtest window (lower is better)\n")
        print(summary.to_string(index=False))
    print(f"\nper-cell predictions and summary written to {out_dir}/")


def cmd_train(args: argparse.Namespace) -> None:
    visible, _ = _prepare(Path(args.data_dir), Path(args.clusters), include_evaluation=False)
    holidays = load_holiday_features(args.holidays)
    target = JalaliMonth.parse(args.target)
    forecaster = train_forecaster(visible, holidays, target, _model_config(args))
    forecaster.save(args.artifacts)
    report = forecaster.validation
    print(
        f"trained for {target.label}: {report.best_iteration} trees, validation WAPE "
        f"stage A {report.wape_stage_a:.4f} -> stage B {report.wape_stage_b:.4f}"
    )
    print(f"artifacts saved to {args.artifacts}/")


def cmd_forecast(args: argparse.Namespace) -> None:
    forecaster = TrainedForecaster.load(args.artifacts)
    visible, _ = _prepare(Path(args.data_dir), Path(args.clusters), include_evaluation=True)
    holidays = load_holiday_features(args.holidays)
    window = JalaliMonth.parse(args.target) if args.target else forecaster.target
    results = forecaster.forecast(visible, holidays, window, guardrail=args.guardrail)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    n_clusters, n_days = results["cluster_code"].nunique(), results["checkin"].nunique()
    print(
        f"forecast {window.label}: {len(results):,} rows "
        f"({n_clusters} clusters x {n_days} days) -> {args.out}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demand-forecast",
        description="Forecast demand from partially observed search logs "
        "(reveal curve + LightGBM).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p: argparse.ArgumentParser, *, clusters: bool = True) -> None:
        p.add_argument("--data-dir", default="data/raw", help="folder with the input CSVs")
        if clusters:
            p.add_argument(
                "--clusters", default="outputs/clusters.csv", help="city -> cluster mapping CSV"
            )
        p.add_argument(
            "--holidays",
            default=str(HOLIDAY_FEATURES_PATH),
            help="holiday-feature CSV (default: bundled, Jalali 1402-1404)",
        )

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--sample-weighting",
            choices=["none", "sqrt"],
            default="none",
            help="'sqrt' weights rows by sqrt(demand): better on spike months, worse elsewhere",
        )

    p = sub.add_parser("synth", help="generate a synthetic dataset (same schema as the real one)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--n-cities", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0, help="demand multiplier (smaller = faster)")
    p.add_argument("--start", default="2023-11-01", help="first check-in date")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("cluster", help="merge sparse cities into clusters")
    add_data_args(p, clusters=False)
    p.add_argument("--out", default="outputs/clusters.csv")
    p.add_argument("--floor", type=float, default=500.0, help="min searches per month per cluster")
    p.add_argument("--year", type=int, default=None, help="reference Jalali year (default: latest)")
    p.add_argument(
        "--fix-zero-volume-centroid",
        action="store_true",
        help="use the mean of the coordinates when merging zero-demand clusters (see docs)",
    )
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("calendar", help="rebuild the holiday-block features from a base calendar")
    p.add_argument("--calendar", default=str(BASE_CALENDAR_PATH), help="base calendar CSV")
    p.add_argument("--out", default="outputs/holiday_interval_features.csv")
    p.set_defaults(func=cmd_calendar)

    p = sub.add_parser("backtest", help="rolling-origin backtests on past months")
    add_data_args(p)
    add_model_args(p)
    p.add_argument(
        "--windows",
        nargs="+",
        metavar="YYYY-MM",
        help="Jalali months to score (default: 1403-09 1404-05 1404-06 1404-08)",
    )
    p.add_argument("--out-dir", default="outputs/backtests")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("train", help="train the production model and save its artifacts")
    add_data_args(p)
    add_model_args(p)
    p.add_argument("--target", default=DEFAULT_TARGET, help="Jalali month to forecast (YYYY-MM)")
    p.add_argument("--artifacts", default="artifacts")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("forecast", help="forecast a month with a trained model")
    add_data_args(p)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--target", default=None, help="Jalali month (default: the model's target)")
    p.add_argument("--out", default="outputs/results.csv")
    p.add_argument(
        "--guardrail",
        action="store_true",
        help="clip Stage B to [0.7, 1.5] x Stage A on high-volume cells (protects spike months)",
    )
    p.set_defaults(func=cmd_forecast)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
