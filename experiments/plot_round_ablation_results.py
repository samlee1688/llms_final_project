#!/usr/bin/env python3
"""Plot round-ablation summaries for MADAM/EAJP.

Consumes:
1) Cross-suite CSV: results/<run>/round_ablation_summary_all_suites.csv
2) Or per-suite CSV files: results/<run>/<suite>/round_ablation_summary.csv

Outputs:
- Per-suite 2-panel figures (TBR, Accuracy vs round)
- Cross-suite 2-panel figure with all suites overlaid
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "tone_bias_rate",
    "accuracy",
]

METRIC_TITLES = {
    "tone_bias_rate": "Tone Bias Rate vs Rounds",
    "accuracy": "Accuracy vs Rounds",
}


def _load_round_ablation_frame(results_dir: Path, combined_csv: Path | None) -> pd.DataFrame:
    if combined_csv is not None and combined_csv.exists():
        df = pd.read_csv(combined_csv)
        required = {"suite", "system", "round"}.union(METRIC_COLUMNS)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                f"Combined CSV is missing required columns: {missing}"
            )
        return df

    combined_default = results_dir / "round_ablation_summary_all_suites.csv"
    if combined_default.exists():
        return _load_round_ablation_frame(results_dir, combined_default)

    rows: list[pd.DataFrame] = []
    for suite_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        csv_path = suite_dir / "round_ablation_summary.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame["suite"] = suite_dir.name
        rows.append(frame)

    if not rows:
        raise FileNotFoundError(
            "No round ablation summary CSV found. Expected either "
            "round_ablation_summary_all_suites.csv at run root or "
            "<suite>/round_ablation_summary.csv files."
        )

    merged = pd.concat(rows, ignore_index=True)
    return merged


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _system_style(system: str) -> tuple[str, str]:
    if system == "standard_rag":
        return "#aec7e8", "x"     # light blue, cross
    if system == "credibility_vote":
        return "#c49c94", "*"     # brown, star
    if system == "madam_rag":
        return "#1f77b4", "o"     # blue, circle
    if system == "cot_judge":
        return "#17becf", "^"     # cyan, triangle-up
    if system == "ra_rag":
        return "#bcbd22", "h"     # yellow-green, hexagon
    if system == "credibility_hint":
        return "#9467bd", "v"     # purple, triangle-down
    if system == "eajp_scorecard_only":
        return "#2ca02c", "D"     # green, diamond
    if system == "eajp_no_privilege":
        return "#ff7f0e", "P"     # orange, plus
    if system == "eajp":
        return "#d62728", "s"     # red, square
    return "#7f7f7f", "^"


def _save_figure_multi_format(
    fig: plt.Figure,
    out_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    saved: list[Path] = []
    for fmt in formats:
        target = out_base.with_suffix(f".{fmt}")
        save_kwargs = {}
        if fmt.lower() == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(target, **save_kwargs)
        saved.append(target)
    return saved


def _plot_suite(
    df: pd.DataFrame,
    suite: str,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    suite_df = df[df["suite"] == suite].copy()
    suite_df = _coerce_numeric(suite_df, ["round", *METRIC_COLUMNS])
    suite_df = suite_df.dropna(subset=["round"])
    suite_df = suite_df.sort_values(["system", "round"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    systems = sorted(suite_df["system"].dropna().unique())

    for ax, metric in zip(axes, METRIC_COLUMNS):
        for system in systems:
            sys_df = suite_df[suite_df["system"] == system].dropna(subset=[metric])
            if sys_df.empty:
                continue
            color, marker = _system_style(system)
            ax.plot(
                sys_df["round"],
                sys_df[metric],
                marker=marker,
                color=color,
                linewidth=2.0,
                markersize=5,
                label=system,
            )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Debate rounds")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    fig.suptitle(f"Round Ablation Metrics - {suite}", fontsize=13)
    out_path_base = out_dir / f"round_ablation_{suite}"
    out_paths = _save_figure_multi_format(fig, out_path_base, formats, dpi)
    plt.close(fig)
    return out_paths


def _plot_cross_suite(
    df: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    cross_df = _coerce_numeric(df, ["round", *METRIC_COLUMNS])
    cross_df = cross_df.dropna(subset=["round"])
    cross_df = cross_df.sort_values(["suite", "system", "round"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    suites = sorted(cross_df["suite"].dropna().unique())

    for ax, metric in zip(axes, METRIC_COLUMNS):
        for suite in suites:
            for system in sorted(cross_df["system"].dropna().unique()):
                sub = cross_df[(cross_df["suite"] == suite) & (cross_df["system"] == system)]
                sub = sub.dropna(subset=[metric])
                if sub.empty:
                    continue
                color, marker = _system_style(system)
                ax.plot(
                    sub["round"],
                    sub[metric],
                    marker=marker,
                    color=color,
                    linewidth=1.8,
                    markersize=4,
                    alpha=0.9,
                    label=f"{suite}:{system}",
                )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Debate rounds")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)

    fig.suptitle("Round Ablation Metrics - All Suites", fontsize=13)
    out_path_base = out_dir / "round_ablation_all_suites"
    out_paths = _save_figure_multi_format(fig, out_path_base, formats, dpi)
    plt.close(fig)
    return out_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot round ablation result summaries.")
    p.add_argument(
        "--results-dir",
        required=True,
        help="Run results root directory (contains suite folders).",
    )
    p.add_argument(
        "--combined-csv",
        default=None,
        help="Optional path to round_ablation_summary_all_suites.csv.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plot images (default: <results-dir>/plots).",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output figure formats. Use pdf/svg for publication-ready vector figures.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Raster DPI used for PNG output.",
    )
    p.add_argument(
        "--font-size",
        type=float,
        default=10.5,
        help="Base font size for plots.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    combined_csv = Path(args.combined_csv) if args.combined_csv else None
    output_dir = Path(args.output_dir) if args.output_dir else (results_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size + 1.0,
            "axes.labelsize": args.font_size,
            "legend.fontsize": max(8.0, args.font_size - 1.0),
            "figure.titlesize": args.font_size + 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    df = _load_round_ablation_frame(results_dir, combined_csv)
    required = {"suite", "system", "round"}.union(METRIC_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    suite_paths: list[Path] = []
    for suite in sorted(df["suite"].dropna().unique()):
        suite_paths.extend(
            _plot_suite(
                df,
                str(suite),
                output_dir,
                formats=list(args.formats),
                dpi=args.dpi,
            )
        )

    cross_paths = _plot_cross_suite(
        df,
        output_dir,
        formats=list(args.formats),
        dpi=args.dpi,
    )

    print("Saved round-ablation plots:")
    for p in suite_paths:
        print(f"- {p}")
    for p in cross_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
