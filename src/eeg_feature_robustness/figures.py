"""Publication figures and summary tables.

All plotting and tabular reporting lives in this single module so the
rendering pipeline is easy to scan. Each ``plot_*`` / ``write_*`` function
takes a results object and an output path; ``generate_figures_and_tables``
is the one entry point that writes every tracked artifact in one pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    FEATURE_COLORS,
    FEATURE_LABELS,
    FEATURE_NAMES,
    MODEL_LABELS,
    MODEL_NAMES,
    TRACKED_RESULT_PATHS,
)

PRIMARY_PROTOCOL_KEY = "pca_matched"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_main_results(results_path: Path):
    """Load the main experiment results JSON."""
    with Path(results_path).open() as f:
        return json.load(f)


def _ensure_parent(output_path: Path) -> Path:
    """Create the parent directory for ``output_path`` if missing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_protocol_results(results, protocol_key=PRIMARY_PROTOCOL_KEY):
    """Return the protocol block used for publication reporting."""
    if results.get("schema_version", 1) >= 3:
        return results[protocol_key]
    return results


def _within_key(protocol_results, dataset):
    grouped_key = f"within_{dataset}_subject_grouped"
    legacy_key = f"within_{dataset}"
    return grouped_key if grouped_key in protocol_results else legacy_key


# ---------------------------------------------------------------------------
# Frame builders (shared with downstream tables)
# ---------------------------------------------------------------------------

def build_within_results_frame(results):
    """Return a tabular representation of within-dataset results."""
    protocol_results = resolve_protocol_results(results)
    phys_key = _within_key(protocol_results, "physionet")
    bci_key = _within_key(protocol_results, "bci2a")
    rows = []
    for feat in FEATURE_NAMES:
        for model in MODEL_NAMES:
            rows.append({
                "Feature": FEATURE_LABELS[feat],
                "Model": MODEL_LABELS[model],
                "PhysioNet": protocol_results[phys_key][feat][model]["mean"],
                "BCI2a": protocol_results[bci_key][feat][model]["mean"],
            })
    return pd.DataFrame(rows)


def build_gap_frame(results):
    """Return a tabular representation of normalised generalisation gaps."""
    protocol_results = resolve_protocol_results(results)
    rows = []
    for feat in FEATURE_NAMES:
        for model in MODEL_NAMES:
            rows.append({
                "Feature": FEATURE_LABELS[feat],
                "Model": MODEL_LABELS[model],
                "Direction": "Phys \u2192 BCI",
                "Norm Gap": protocol_results["gaps"][feat][model]["phys_to_bci"]["normalised_gap"],
            })
            rows.append({
                "Feature": FEATURE_LABELS[feat],
                "Model": MODEL_LABELS[model],
                "Direction": "BCI \u2192 Phys",
                "Norm Gap": protocol_results["gaps"][feat][model]["bci_to_phys"]["normalised_gap"],
            })
    return pd.DataFrame(rows)


def build_scatter_frame(results):
    """Return a tabular representation of within- vs cross-dataset accuracy."""
    protocol_results = resolve_protocol_results(results)
    phys_key = _within_key(protocol_results, "physionet")
    bci_key = _within_key(protocol_results, "bci2a")
    rows = []
    for feat in FEATURE_NAMES:
        for model in MODEL_NAMES:
            rows.append({
                "Feature": feat,
                "Model": model,
                "Direction": "Phys\u2192BCI",
                "Within": protocol_results[phys_key][feat][model]["mean"],
                "Cross": protocol_results["cross_phys_to_bci"][feat][model]["balanced_accuracy"],
            })
            rows.append({
                "Feature": feat,
                "Model": model,
                "Direction": "BCI\u2192Phys",
                "Within": protocol_results[bci_key][feat][model]["mean"],
                "Cross": protocol_results["cross_bci_to_phys"][feat][model]["balanced_accuracy"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main-experiment figures
# ---------------------------------------------------------------------------

def plot_within_dataset_heatmap(results, output_path: Path):
    """Generate the within-dataset heatmap."""
    output_path = _ensure_parent(output_path)
    df_within = build_within_results_frame(results)
    pivot = df_within.set_index(["Feature", "Model"])[["PhysioNet", "BCI2a"]]

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.50,
        vmax=0.70,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"label": "Balanced accuracy", "shrink": 0.85},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_generalisation_gap(results, output_path: Path):
    """Generate the normalised generalisation-gap bar figure.

    Averages gaps across all classical models. No feature-model cell is removed;
    if a source baseline is near chance, the paper discusses that limitation
    explicitly rather than filtering it after the fact.
    """
    output_path = _ensure_parent(output_path)
    df_gaps = build_gap_frame(results)
    df_avg = (
        df_gaps
        .groupby(["Feature", "Direction"])["Norm Gap"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(FEATURE_NAMES))
    width = 0.36
    directions = ["Phys \u2192 BCI", "BCI \u2192 Phys"]
    direction_colors = {"Phys \u2192 BCI": "#4C78A8", "BCI \u2192 Phys": "#E6872E"}

    vals_min = 0.0
    vals_max = 0.0
    for i, direction in enumerate(directions):
        vals = [
            df_avg[
                (df_avg["Feature"] == FEATURE_LABELS[f]) &
                (df_avg["Direction"] == direction)
            ]["Norm Gap"].values[0]
            for f in FEATURE_NAMES
        ]
        bars = ax.bar(
            x + i * width - width / 2,
            vals,
            width,
            label=direction,
            color=direction_colors[direction],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            offset = 0.006 if val >= 0 else -0.006
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:+.3f}",
                ha="center",
                va=va,
                fontsize=10,
                fontweight="semibold",
                color="#222222",
            )
        vals_min = min(vals_min, min(vals))
        vals_max = max(vals_max, max(vals))

    ax.axhline(0, color="black", linewidth=0.9, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], fontsize=12)
    ax.set_ylabel("Mean normalised generalisation gap")
    ax.set_ylim(vals_min - 0.04, vals_max + 0.05)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_within_vs_cross(results, output_path: Path):
    """Generate the within- vs cross-dataset scatter figure."""
    output_path = _ensure_parent(output_path)
    df_scatter = build_scatter_frame(results)
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    markers = {"Phys\u2192BCI": "o", "BCI\u2192Phys": "^"}

    for feat in FEATURE_NAMES:
        for direction, marker in markers.items():
            subset = df_scatter[
                (df_scatter["Feature"] == feat) &
                (df_scatter["Direction"] == direction)
            ]
            ax.scatter(
                subset["Within"],
                subset["Cross"],
                color=FEATURE_COLORS[feat],
                marker=marker,
                s=110,
                alpha=0.9,
                linewidth=0.8,
                edgecolor="white",
                zorder=3,
            )

    lims = [0.48, 0.72]
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.1, alpha=0.7,
            label="No degradation (y = x)", zorder=2)
    ax.fill_between(lims, [0.48, 0.48], lims, alpha=0.07, color="#C0392B", zorder=1)
    ax.text(0.693, 0.495, "Degradation\nzone", fontsize=9, color="#8B2E1F",
            ha="right", va="bottom", alpha=0.85)

    feature_patches = [
        mpatches.Patch(color=FEATURE_COLORS[f], label=FEATURE_LABELS[f])
        for f in FEATURE_NAMES
    ]
    direction_handles = [
        plt.scatter([], [], marker="o", color="grey", s=70, label="Phys $\\to$ BCI"),
        plt.scatter([], [], marker="^", color="grey", s=70, label="BCI $\\to$ Phys"),
    ]
    ax.legend(
        handles=feature_patches + direction_handles,
        fontsize=10,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
    )
    ax.set_xlabel("Within-dataset balanced accuracy")
    ax.set_ylabel("Cross-dataset balanced accuracy")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Per-subject figures
# ---------------------------------------------------------------------------

def plot_bci2a_per_subject(per_subject_results, output_path: Path):
    """Per-subject BCI2a balanced-accuracy boxplot, faceted by classifier."""
    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Per-Subject BCI2a Balanced Accuracy (5-fold CV within each subject)\n"
        "Chance level = 0.50",
        fontsize=12, y=1.02,
    )

    for ax, model_name in zip(axes, MODEL_NAMES):
        data = [per_subject_results[feat][model_name]["per_subject"] for feat in FEATURE_NAMES]
        labels = [FEATURE_LABELS[f] for f in FEATURE_NAMES]
        colors = [FEATURE_COLORS[f] for f in FEATURE_NAMES]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=5, alpha=0.6),
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.axhline(0.5, color="black", linestyle="--",
                   linewidth=1, alpha=0.5, label="Chance")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(MODEL_LABELS[model_name], fontsize=11)
        ax.set_ylim(0.40, 0.90)
        ax.grid(axis="y", alpha=0.3)

        if ax is axes[0]:
            ax.set_ylabel("Balanced Accuracy", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined_per_subject(bci_results, physionet_results, output_path: Path):
    """Combined per-subject boxplot for BCI2a and PhysioNet."""
    output_path = _ensure_parent(output_path)

    bci_data = [bci_results[feat]["LogReg"]["per_subject"] for feat in FEATURE_NAMES]
    phys_data = [
        [value["mean"] for value in physionet_results[feat].values() if value is not None]
        for feat in FEATURE_NAMES
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True,
                             constrained_layout=True)

    panels = [
        (axes[0], bci_data, "BCI2a ($n{=}9$ subjects)"),
        (axes[1], phys_data, "PhysioNet ($n{=}109$ subjects)"),
    ]

    rng = np.random.default_rng(0)
    for ax, data, title in panels:
        boxplot = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(linewidth=1.3, color="#333333"),
            capprops=dict(linewidth=1.3, color="#333333"),
            boxprops=dict(linewidth=1.0, edgecolor="#333333"),
            flierprops=dict(marker="", markersize=0),
            widths=0.55,
        )
        for patch, feat in zip(boxplot["boxes"], FEATURE_NAMES):
            patch.set_facecolor(FEATURE_COLORS[feat])
            patch.set_alpha(0.55)

        for i, feat_vals in enumerate(data, start=1):
            xs = rng.normal(loc=i, scale=0.06, size=len(feat_vals))
            ax.scatter(
                xs,
                feat_vals,
                s=18,
                alpha=0.55,
                color=FEATURE_COLORS[FEATURE_NAMES[i - 1]],
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES])
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylim(0.20, 0.95)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("Balanced accuracy (within-subject 5-fold CV)")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_within_dataset_table(results, output_path: Path):
    """Write the within-dataset summary table."""
    output_path = _ensure_parent(output_path)
    protocol_results = resolve_protocol_results(results)
    phys_key = _within_key(protocol_results, "physionet")
    bci_key = _within_key(protocol_results, "bci2a")
    print("\n" + "=" * 70)
    print("TABLE 1 - Subject-Grouped Within-Dataset Balanced Accuracy")
    print("=" * 70)
    print(f"{'Feature':<12} {'Model':<14} {'PhysioNet':>16} {'BCI2a':>16}")
    print("-" * 70)

    rows = []
    for feat in FEATURE_NAMES:
        for model in MODEL_NAMES:
            p_result = protocol_results[phys_key][feat][model]
            b_result = protocol_results[bci_key][feat][model]
            p_mean = p_result["mean"]
            p_std = p_result["std"]
            b_mean = b_result["mean"]
            b_std = b_result["std"]
            print(
                f"{FEATURE_LABELS[feat]:<12} {MODEL_LABELS[model]:<14} "
                f"{p_mean:.3f} \u00b1 {p_std:.3f}   {b_mean:.3f} \u00b1 {b_std:.3f}"
            )
            rows.append({
                "Feature": FEATURE_LABELS[feat],
                "Model": MODEL_LABELS[model],
                "PhysioNet mean": round(p_mean, 4),
                "PhysioNet std": round(p_std, 4),
                "PhysioNet ci95 low": p_result.get("ci95", {}).get("low"),
                "PhysioNet ci95 high": p_result.get("ci95", {}).get("high"),
                "BCI2a mean": round(b_mean, 4),
                "BCI2a std": round(b_std, 4),
                "BCI2a ci95 low": b_result.get("ci95", {}).get("low"),
                "BCI2a ci95 high": b_result.get("ci95", {}).get("high"),
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


def write_cross_dataset_table(results, output_path: Path):
    """Write the cross-dataset summary table."""
    output_path = _ensure_parent(output_path)
    protocol_results = resolve_protocol_results(results)
    print("\n" + "=" * 90)
    print("TABLE 2 - Cross-Dataset Accuracy and Normalised Generalisation Gap")
    print("=" * 90)
    print(
        f"{'Feature':<12} {'Model':<14} {'P\u2192B acc':>9} {'P\u2192B gap':>9} "
        f"{'B\u2192P acc':>9} {'B\u2192P gap':>9}"
    )
    print("-" * 90)

    rows = []
    for feat in FEATURE_NAMES:
        for model in MODEL_NAMES:
            p2b_result = protocol_results["cross_phys_to_bci"][feat][model]
            b2p_result = protocol_results["cross_bci_to_phys"][feat][model]
            p2b_acc = p2b_result["balanced_accuracy"]
            b2p_acc = b2p_result["balanced_accuracy"]
            p2b_gap = protocol_results["gaps"][feat][model]["phys_to_bci"]["normalised_gap"]
            b2p_gap = protocol_results["gaps"][feat][model]["bci_to_phys"]["normalised_gap"]
            print(
                f"{FEATURE_LABELS[feat]:<12} {MODEL_LABELS[model]:<14} "
                f"{p2b_acc:>9.3f} {p2b_gap:>9.3f} {b2p_acc:>9.3f} {b2p_gap:>9.3f}"
            )
            rows.append({
                "Feature": FEATURE_LABELS[feat],
                "Model": MODEL_LABELS[model],
                "Phys\u2192BCI accuracy": round(p2b_acc, 4),
                "Phys\u2192BCI ci95 low": p2b_result.get("ci95", {}).get("low"),
                "Phys\u2192BCI ci95 high": p2b_result.get("ci95", {}).get("high"),
                "Phys\u2192BCI norm gap": round(p2b_gap, 4),
                "BCI\u2192Phys accuracy": round(b2p_acc, 4),
                "BCI\u2192Phys ci95 low": b2p_result.get("ci95", {}).get("low"),
                "BCI\u2192Phys ci95 high": b2p_result.get("ci95", {}).get("high"),
                "BCI\u2192Phys norm gap": round(b2p_gap, 4),
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


def print_written_analysis(results):
    """Print a compact written interpretation of the results."""
    protocol_results = resolve_protocol_results(results)
    print("\n" + "=" * 70)
    print("WRITTEN ANALYSIS")
    print("=" * 70)

    avg_gaps = {}
    for feat in FEATURE_NAMES:
        all_norm_gaps = [
            protocol_results["gaps"][feat][model][direction]["normalised_gap"]
            for model in MODEL_NAMES
            for direction in ["phys_to_bci", "bci_to_phys"]
        ]
        avg_gaps[feat] = float(np.mean(all_norm_gaps))

    most_robust = min(avg_gaps, key=avg_gaps.get)
    least_robust = max(avg_gaps, key=avg_gaps.get)

    print(
        f"\n1. Smallest descriptive mean gap: {FEATURE_LABELS[most_robust]} "
        f"(avg normalised gap: {avg_gaps[most_robust]:.4f})"
    )
    print(
        f"   Least robust: {FEATURE_LABELS[least_robust]} "
        f"(avg normalised gap: {avg_gaps[least_robust]:.4f})"
    )

    print("\n2. Treat feature rankings as descriptive unless the statistical")
    print("   summary supports a stable pairwise difference.")
    print("   The paper should emphasize that no classical representation solved")
    print("   zero-shot cross-dataset transfer.")

    model_avg_gaps = {}
    for model in MODEL_NAMES:
        all_model_gaps = [
            protocol_results["gaps"][feat][model][direction]["normalised_gap"]
            for feat in FEATURE_NAMES
            for direction in ["phys_to_bci", "bci_to_phys"]
        ]
        model_avg_gaps[model] = float(np.mean(all_model_gaps))

    most_robust_model = min(model_avg_gaps, key=model_avg_gaps.get)
    print(
        f"\n3. Smallest descriptive model gap: {MODEL_LABELS[most_robust_model]} "
        f"(avg normalised gap: {model_avg_gaps[most_robust_model]:.4f})"
    )

    print("\n4. Trial-wise CV is diagnostic only. Subject-grouped estimates are the")
    print("   primary within-dataset baseline for publication claims.")


# ---------------------------------------------------------------------------
# One-shot orchestrator
# ---------------------------------------------------------------------------

def generate_figures_and_tables(
    main_results=None,
    bci_per_subject=None,
    physionet_per_subject=None,
):
    """Render every tracked figure and table.

    Each result object is loaded from disk on demand if not supplied. Per-subject
    figures are skipped when the corresponding inputs are absent so this entry
    point can run in any order against partially-populated `results/`.
    """
    if main_results is None:
        main_results = load_main_results(TRACKED_RESULT_PATHS["main_experiment_results"])

    plot_within_dataset_heatmap(main_results, TRACKED_RESULT_PATHS["within_dataset_accuracy_heatmap"])
    plot_generalisation_gap(main_results, TRACKED_RESULT_PATHS["generalisation_gap_by_direction"])
    plot_within_vs_cross(main_results, TRACKED_RESULT_PATHS["within_vs_cross_dataset_accuracy"])
    write_within_dataset_table(main_results, TRACKED_RESULT_PATHS["within_dataset_results"])
    write_cross_dataset_table(main_results, TRACKED_RESULT_PATHS["cross_dataset_results"])
    print_written_analysis(main_results)

    bci_path = TRACKED_RESULT_PATHS["bci2a_per_subject_results"]
    if bci_per_subject is None and bci_path.exists():
        with bci_path.open() as f:
            bci_per_subject = json.load(f)
    if bci_per_subject is not None:
        plot_bci2a_per_subject(bci_per_subject, TRACKED_RESULT_PATHS["bci2a_per_subject_accuracy"])

    phys_path = TRACKED_RESULT_PATHS["physionet_subject_results"]
    if physionet_per_subject is None and phys_path.exists():
        with phys_path.open() as f:
            physionet_per_subject = json.load(f)
    if bci_per_subject is not None and physionet_per_subject is not None:
        plot_combined_per_subject(
            bci_per_subject,
            physionet_per_subject,
            TRACKED_RESULT_PATHS["combined_per_subject_accuracy"],
        )


def main():
    """CLI entrypoint for figure/table generation."""
    generate_figures_and_tables()


if __name__ == "__main__":
    main()
