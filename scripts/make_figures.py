"""Render the README figures from backtest outputs and a trained model's artifacts.

    python scripts/make_figures.py \
        --backtests outputs/backtests --artifacts artifacts --out docs/images

Inputs are exactly what ``demand-forecast backtest`` and ``demand-forecast train`` write, so the
figures can be regenerated for any dataset (including the synthetic one). Requires matplotlib
(``pip install -e ".[plots]"``).

Design: one colour per *method*, fixed across every figure (gray = naive, orange = Stage A,
light blue = Stage B before calibration, blue = final calibrated model, ink = ground truth);
thin marks, hairline solid grid, legend whenever there are >= 2 series, values labelled
sparingly. Palette: the documented categorical slots 1-2 plus a tint of slot 1.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from demand_forecasting.jalali import JalaliMonth
from demand_forecasting.metrics import wape

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

COLORS = {
    "naive": "#a8a69e",  # de-emphasised baseline
    "stage_a": "#eb6834",  # categorical slot 2
    "stage_b": "#86b6ef",  # tint of slot 1 (same model as the final one, before calibration)
    "final": "#2a78d6",  # categorical slot 1: the shipped configuration
    "truth": INK,
}
# Emphasis form for the weekday chart: one accent hue + neutral context, deliberately NOT the
# method colours above (colour follows the entity: these series are weekday groups).
WEEKEND_ACCENT, WEEKDAY_CONTEXT = "#4a3aa7", "#898781"
LABELS = {
    "naive": "Naive (pooled curve)",
    "stage_a": "Stage A (reveal curve)",
    "stage_b": "Stage B",
    "final": "Stage B + calibration",
    "truth": "Actual",
}
PRED_COLUMNS = {
    "naive": "pred_naive",
    "stage_a": "pred_stage_a",
    "stage_b": "pred_stage_b",
    "final": "pred_stage_b_cal_lead",
}
LEAD_BUCKETS = [(0, 3), (3, 7), (7, 14), (14, 21), (21, 30)]
LEAD_TICKS = ["1-3", "4-7", "8-14", "15-21", "22-30"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.grid.axis": "y",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 9.5,
        }
    )


def _header(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=INK_2, va="top")


def _load_backtests(directory: Path) -> dict[JalaliMonth, pd.DataFrame]:
    frames = {}
    for path in sorted(directory.glob("backtest_*.csv")):
        match = re.fullmatch(r"backtest_(\d{4}-\d{2})\.csv", path.name)
        if match:
            frames[JalaliMonth.parse(match.group(1))] = pd.read_csv(path, parse_dates=["checkin"])
    if not frames:
        raise SystemExit(f"no backtest_YYYY-MM.csv files in {directory}")
    return dict(sorted(frames.items(), key=lambda kv: (kv[0].year, kv[0].month)))


def _metrics(frames: dict[JalaliMonth, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for window, frame in frames.items():
        row = {"window": window.label}
        for key, column in PRED_COLUMNS.items():
            row[key] = wape(frame["true_total"], frame[column])
        rows.append(row)
    return pd.DataFrame(rows)


def fig_wape_by_window(frames, out: Path) -> None:
    table = _metrics(frames)
    methods = ["naive", "stage_a", "stage_b", "final"]
    n_windows, slot = len(table), 0.8
    width = slot / len(methods) * 0.94  # the leftover is the ~2px surface gap between bars
    x = np.arange(n_windows)

    fig, ax = plt.subplots(figsize=(9, 4.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.12)
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * slot / len(methods)
        bars = ax.bar(
            x + offset, table[method], width=width, color=COLORS[method], label=LABELS[method]
        )
        if method == "final":  # label only the series the story is about
            for bar, value in zip(bars, table[method], strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=INK_2,
                )
    ax.set_xticks(x, table["window"])
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_ylim(0, max(table[methods].max()) * 1.12)
    ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=INK_2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    worst = table["final"].idxmax()
    if table.loc[worst, "stage_b"] > table.loc[worst, "stage_a"] * 1.5:
        ax.annotate(
            "seasonal-spike month:\nStage B under-predicts the peak",
            xy=(worst, table.loc[worst, "stage_b"] + 0.045),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK_2,
        )
    handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.855), ncol=4, fontsize=9)
    _header(
        fig,
        "Forecast error on held-out months",
        "WAPE of forecasts for every day of a month the model never saw "
        "(made the day before the month starts)",
    )
    fig.savefig(out / "wape_by_window.png", dpi=200)
    plt.close(fig)


def fig_error_by_lead(frames, out: Path) -> None:
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.6, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.70, bottom=0.17, wspace=0.12)
    top = 0.0
    for ax, (window, frame) in zip(axes, frames.items(), strict=True):
        buckets = pd.cut(frame["lead"], [0, *[hi for _, hi in LEAD_BUCKETS]])
        for key in ("stage_a", "final"):
            values = [
                wape(part["true_total"], part[PRED_COLUMNS[key]])
                for _, part in frame.groupby(buckets, observed=True)
            ]
            top = max(top, max(values))
            ax.plot(
                range(len(values)),
                values,
                color=COLORS[key],
                lw=2,
                marker="o",
                ms=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
        ax.set_title(window.label, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xticks(range(len(LEAD_TICKS)), LEAD_TICKS)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("WAPE (lower is better)")
    for ax in axes:
        ax.set_ylim(0, top * 1.1)
        ax.set_xlim(-0.3, len(LEAD_TICKS) - 0.7)
    fig.text(
        0.5,
        0.035,
        "Forecast horizon: days between the forecast date and the check-in",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )
    handles = [
        Line2D([0], [0], color=COLORS[k], lw=2, marker="o", ms=5.5, label=LABELS[k])
        for k in ("stage_a", "final")
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.855), ncol=2, fontsize=9)
    _header(
        fig,
        "Stage B pays off at long horizons",
        "WAPE by forecast horizon. In the last week Stage A is already accurate (little demand is "
        "still unseen); far-out dates are where the correction helps",
    )
    fig.savefig(out / "error_by_lead.png", dpi=200)
    plt.close(fig)


def fig_forecast_vs_actual(frames, out: Path) -> None:
    metrics = _metrics(frames).set_index("window")["final"]
    labels = {w.label: w for w in frames}
    chosen = [metrics.idxmin()]
    if len(metrics) > 1 and metrics.idxmax() != chosen[0]:
        chosen.append(metrics.idxmax())
    fig, axes = plt.subplots(1, len(chosen), figsize=(5.2 * len(chosen) + 0.4, 4.3), sharey=False)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.14, wspace=0.18)
    for ax, label in zip(axes, chosen, strict=True):
        frame = frames[labels[label]]
        daily = frame.groupby("checkin")[
            ["true_total", "pred_stage_a", "pred_stage_b_cal_lead"]
        ].sum()
        index = 100 / daily["true_total"].mean()
        ax.plot(daily.index, daily["true_total"] * index, color=COLORS["truth"], lw=2.2)
        ax.plot(daily.index, daily["pred_stage_a"] * index, color=COLORS["stage_a"], lw=2)
        ax.plot(daily.index, daily["pred_stage_b_cal_lead"] * index, color=COLORS["final"], lw=2)
        ax.set_title(
            f"{label}  -  final WAPE {metrics[label]:.3f}",
            fontsize=10,
            color=INK,
            loc="left",
            pad=8,
        )
        ticks = daily.index[::7]
        ax.set_xticks(ticks, [t.strftime("%b %d") for t in ticks])
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Daily total demand (index, month mean = 100)")
    handles = [
        Line2D([0], [0], color=COLORS[k], lw=2.2 if k == "truth" else 2, label=LABELS[k])
        for k in ("truth", "stage_a", "final")
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.855), ncol=3, fontsize=9)
    _header(
        fig,
        "Forecast vs. actual: where it works and where it fails",
        "All clusters summed per check-in date; forecast made the day before the month starts",
    )
    fig.savefig(out / "forecast_vs_actual.png", dpi=200)
    plt.close(fig)


def fig_reveal_curve(artifacts: Path, out: Path) -> None:
    global_curve = pd.read_csv(artifacts / "reveal_global.csv", index_col=0)["value"]
    adjustment = pd.read_csv(artifacts / "reveal_dow_adjustment.csv", index_col=0)
    leads = global_curve.index.astype(int).to_numpy()
    by_weekday = adjustment.mul(global_curve.to_numpy(), axis=1).clip(upper=1.0)
    weekend = by_weekday.loc[[3, 4]].mean()  # pandas weekday: Thursday=3, Friday=4
    others = by_weekday.loc[[0, 1, 2, 5, 6]].mean()
    horizon = 45
    mask = leads <= horizon

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.14)
    for series, color, text in (
        (others, WEEKDAY_CONTEXT, "Saturday-Wednesday"),
        (weekend, WEEKEND_ACCENT, "Thursday-Friday (Iran's weekend)"),
    ):
        ax.plot(leads[mask], series.to_numpy()[mask] * 100, color=color, lw=2, label=text)
    at30 = float(weekend.iloc[30] * 100), float(others.iloc[30] * 100)
    ax.scatter(
        [30, 30],
        at30,
        s=36,
        color=[WEEKEND_ACCENT, WEEKDAY_CONTEXT],
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
    )
    ax.annotate(
        f"30 days ahead only {at30[0]:.0f}-{at30[1]:.0f}% of the final\n"
        "demand has been searched yet",
        xy=(30, max(at30)),
        xytext=(36.5, 33),
        fontsize=9,
        color=INK_2,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1},
    )
    ax.set_xlim(horizon, 0)  # time flows towards the check-in on the right
    ax.set_ylim(0, 102)
    ax.set_xlabel("Days before check-in")
    ax.set_ylabel("Share of final demand already searched (%)")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        title="Check-in weekday",
        title_fontsize=9,
        ncol=1,
    )
    _header(
        fig,
        "Demand is right-censored: the reveal curve",
        "Median share of a check-in's final demand logged by N days ahead; weekend check-ins "
        "are planned further in advance",
    )
    fig.savefig(out / "reveal_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backtests", default="outputs/backtests", type=Path)
    parser.add_argument("--artifacts", default="artifacts", type=Path)
    parser.add_argument("--out", default="docs/images", type=Path)
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = _load_backtests(args.backtests)
    fig_wape_by_window(frames, args.out)
    fig_error_by_lead(frames, args.out)
    fig_forecast_vs_actual(frames, args.out)
    if (args.artifacts / "reveal_global.csv").exists():
        fig_reveal_curve(args.artifacts, args.out)
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
