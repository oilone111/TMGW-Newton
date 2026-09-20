from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLORS = {
    "S0": "#4B5563", "S1": "#3B82F6", "P0": "#059669",
    "S2": "#F59E0B", "S3": "#B91C1C",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.015, label, transform=ax.transAxes, weight="bold", fontsize=10,
            ha="left", va="bottom")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training(manifest_path: Path, history_path: Path, output: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    required = {"split", "porosity", "permeability_m2", "aperture_m",
                "surface_density_um_cm2", "penetration_ratio"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.semilogy(epochs, history["train"], color="#315A7D", lw=1.5, label="Training")
    ax.semilogy(epochs, history["validation"], color="#B5483A", lw=1.5, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Composite loss", title="Training convergence")
    handles, labels = ax.get_legend_handles_labels()
    true_ratio = history.get("validation_true_residual_ratio")
    if true_ratio and len(true_ratio) == len(epochs):
        ax2 = ax.twinx()
        ratio = np.asarray(true_ratio, dtype=float)
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        residual_line, = ax2.plot(
            epochs, ratio, color="#238B45", lw=1.15, label="True residual ratio"
        )
        ax2.axhline(1.0, color="#238B45", ls="--", lw=0.75, alpha=0.55)
        ax2.set_ylabel(r"$\rho_R$ (complete equations)", color="#238B45")
        ax2.tick_params(axis="y", colors="#238B45")
        handles.append(residual_line)
        labels.append("True residual ratio")
    ax.legend(handles, labels, ncol=1, fontsize=7.5, loc="best")
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    split_order = [x for x in ("train", "validation", "test") if x in set(manifest["split"])]
    counts = manifest["split"].value_counts().reindex(split_order)
    bars = ax.bar(split_order, counts, color=["#315A7D", "#D49A3A", "#6B7280"][:len(counts)], width=0.62)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set(ylabel="Number of complete cases", title="Case-wise dataset split")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    markers = {"train": "o", "validation": "s", "test": "^"}
    split_colors = {"train": "#315A7D", "validation": "#D49A3A", "test": "#6B7280"}
    for split, group in manifest.groupby("split"):
        ax.scatter(group["porosity"], group["permeability_m2"] / 1e-15,
                   s=24, marker=markers.get(split, "o"), color=split_colors.get(split, "#555555"),
                   edgecolor="white", linewidth=0.35, label=split.capitalize())
    ax.set(xlabel="Porosity", ylabel=r"Permeability ($10^{-15}$ m$^2$)",
           title="Matrix properties")
    ax.legend(ncol=3, fontsize=8)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    sc = ax.scatter(manifest["aperture_m"] * 1e6, manifest["surface_density_um_cm2"],
                    c=manifest["penetration_ratio"], cmap="viridis", s=28,
                    edgecolor="white", linewidth=0.35)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Penetration ratio")
    ax.set(xlabel=r"Microfracture aperture ($\mu$m)",
           ylabel=r"Surface density ($\mu$m cm$^{-2}$)",
           title="Microfracture parameters")
    panel_label(ax, "(d)")
    save_figure(fig, output)


def read_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for mode in ("S0", "S1"):
        if mode not in data:
            raise ValueError(f"benchmark summary must contain {mode}")
    return data


def plot_efficiency(benchmark_dir: Path, output: Path) -> None:
    summary = read_summary(benchmark_dir / "summary.json")
    modes = [mode for mode in ("S0", "S1", "P0", "S2", "S3") if mode in summary]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)

    ax = axes[0, 0]
    for mode in modes:
        history = pd.read_csv(benchmark_dir / mode / "newton_history.csv")
        if history.empty:
            continue
        selected_time = history.groupby("time_s")["iteration"].max().idxmax()
        selected = history[history["time_s"] == selected_time]
        linestyle = {"S0": "-", "S1": "--", "P0": "-.", "S2": (0, (3, 1, 1, 1)), "S3": ":"}[mode]
        ax.semilogy(selected["iteration"], selected["residual_norm"], marker="o", ms=3,
                    lw=1.4, ls=linestyle, color=COLORS[mode], label=mode)
    ax.set(xlabel="Newton iteration", ylabel=r"$\|R\|_\infty$",
           title="Residual convergence")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=len(modes))
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    values = [summary[m]["mean_newton_iterations"] for m in modes]
    bars = ax.bar(modes, values, color=[COLORS[m] for m in modes], width=0.62)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set(ylabel="Mean Newton iterations", title="Newton iterations")
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    values = [summary[m]["online_time_s"] for m in modes]
    bars = ax.bar(modes, values, color=[COLORS[m] for m in modes], width=0.62)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set(ylabel="Online time (s)", title="Computation time")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    speedup = [summary[m]["speedup_vs_S0"] for m in modes]
    rollback = [summary[m]["rollbacks"] for m in modes]
    x = np.arange(len(modes))
    bars = ax.bar(x - 0.18, speedup, width=0.34, color=[COLORS[m] for m in modes], label="Speedup")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, rollback, color="#111827", marker="D", ms=4, lw=1.2, label="Rollbacks")
    ax.set_xticks(x, modes)
    ax.set_ylabel("Speedup vs. S0")
    ax2.set_ylabel("Rollback count")
    ax2.set_ylim(-0.1, max(float(max(rollback)) * 1.25, 1.0))
    ax.set_title("Speedup and robustness")
    ax.axhline(1.0, color="#9CA3AF", ls="--", lw=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    panel_label(ax, "(d)")
    save_figure(fig, output)


def plot_accuracy(benchmark_dir: Path, ablation_path: Path, output: Path) -> None:
    summary = read_summary(benchmark_dir / "summary.json")
    modes = [mode for mode in ("S1", "P0", "S2", "S3") if mode in summary]
    ablation = pd.read_csv(ablation_path)
    required = {"variant", "mean_newton_iterations", "online_time_s", "fallback_rate"}
    if required - set(ablation.columns):
        raise ValueError("ablation table does not satisfy the required data contract")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)

    ax = axes[0]
    metrics = ["pressure_relative_l2", "water_saturation_mae", "gas_saturation_mae"]
    labels = [r"$p$ relative $L_2$", r"$S_w$ MAE", r"$S_g$ MAE"]
    x = np.arange(len(metrics))
    width = 0.72 / max(len(modes), 1)
    for k, mode in enumerate(modes):
        ax.bar(x + (k - (len(modes) - 1) / 2) * width,
               [max(summary[mode][metric], 1e-16) for metric in metrics],
               width=width, color=COLORS[mode], label=mode)
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Final-solution consistency")
    ax.legend(ncol=len(modes))
    panel_label(ax, "(a)")

    ax = axes[1]
    values = [summary[m]["max_mass_error"] for m in modes]
    bars = ax.bar(modes, values, color=[COLORS[m] for m in modes], width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("Maximum mass error")
    ax.set_title("Three-phase conservation")
    ax.bar_label(bars, fmt="%.1e", padding=2, fontsize=7)
    panel_label(ax, "(b)")

    ax = axes[2]
    x = np.arange(len(ablation))
    ax.plot(x, ablation["mean_newton_iterations"], color="#315A7D", marker="o", label="Newton")
    ax.plot(x, ablation["online_time_s"], color="#B5483A", marker="s", label="Time")
    ax.set_xticks(x, ablation["variant"], rotation=25, ha="right")
    ax.set_ylabel("Normalized/absolute value")
    ax.set_title("Module ablation")
    ax2 = ax.twinx()
    ax2.bar(x, ablation["fallback_rate"], width=0.5, color="#D1D5DB", alpha=0.45)
    ax2.set_ylabel("Fallback rate")
    ax.legend(loc="upper left")
    panel_label(ax, "(c)")
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 4 figures from real logs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/chapter4_figures_v18"))
    args = parser.parse_args()
    set_style()
    plot_training(args.dataset / "dataset_manifest.csv", args.history, args.output / "Fig4-4.png")
    plot_efficiency(args.benchmark, args.output / "Fig4-5.png")
    plot_accuracy(args.benchmark, args.ablation, args.output / "Fig4-6.png")


if __name__ == "__main__":
    main()
