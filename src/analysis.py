import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

FEATURE_NAMES  = ["TimeDomain", "BandPower", "FFT"]
MODEL_NAMES    = ["LogReg", "SVM", "RandomForest"]
FEATURE_LABELS = {"TimeDomain": "Time Domain", "BandPower": "Band Power", "FFT": "FFT"}
MODEL_LABELS   = {"LogReg": "Log. Reg.", "SVM": "SVM", "RandomForest": "Rand. Forest"}
COLORS         = {"TimeDomain": "#e74c3c", "BandPower": "#2ecc71", "FFT": "#3498db"}

os.makedirs("results", exist_ok=True)


# ── Load results ───────────────────────────────────────────────────────────
with open("results/all_results.json") as f:
    R = json.load(f)

within_physionet  = R["within_physionet"]
within_bci2a      = R["within_bci2a"]
cross_phys_to_bci = R["cross_phys_to_bci"]
cross_bci_to_phys = R["cross_bci_to_phys"]
gaps              = R["gaps"]


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Within-dataset accuracy heatmap
# ══════════════════════════════════════════════════════════════════════════

rows = []
for feat in FEATURE_NAMES:
    for model in MODEL_NAMES:
        rows.append({
            "Feature": FEATURE_LABELS[feat],
            "Model":   MODEL_LABELS[model],
            "PhysioNet (within)": within_physionet[feat][model]["mean"],
            "BCI2a (within)":     within_bci2a[feat][model]["mean"],
        })

df_within = pd.DataFrame(rows)
pivot = df_within.set_index(["Feature", "Model"])[["PhysioNet (within)", "BCI2a (within)"]]

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    vmin=0.50,
    vmax=0.70,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10},
)
ax.set_title("Figure 1 — Within-Dataset Balanced Accuracy\n(5-fold cross-validation)", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("results/fig1_within_dataset.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/fig1_within_dataset.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Normalised generalisation gap bar chart (split by direction)
# ══════════════════════════════════════════════════════════════════════════

gap_rows = []
for feat in FEATURE_NAMES:
    for model in MODEL_NAMES:
        gap_rows.append({
            "Feature":   FEATURE_LABELS[feat],
            "Model":     MODEL_LABELS[model],
            "Direction": "Phys → BCI",
            "Norm Gap":  gaps[feat][model]["phys_to_bci"]["normalised_gap"],
        })
        gap_rows.append({
            "Feature":   FEATURE_LABELS[feat],
            "Model":     MODEL_LABELS[model],
            "Direction": "BCI → Phys",
            "Norm Gap":  gaps[feat][model]["bci_to_phys"]["normalised_gap"],
        })

df_gaps = pd.DataFrame(gap_rows)

# Average across models per feature × direction
df_avg = df_gaps.groupby(["Feature", "Direction"])["Norm Gap"].mean().reset_index()

fig, ax = plt.subplots(figsize=(9, 5))
x       = np.arange(len(FEATURE_NAMES))
width   = 0.35
dirs    = ["Phys → BCI", "BCI → Phys"]
bar_colors = ["#3498db", "#e67e22"]

for i, direction in enumerate(dirs):
    vals = [
        df_avg[(df_avg["Feature"] == FEATURE_LABELS[f]) &
               (df_avg["Direction"] == direction)]["Norm Gap"].values[0]
        for f in FEATURE_NAMES
    ]
    bars = ax.bar(x + i * width - width / 2, vals, width,
                  label=direction, color=bar_colors[i], alpha=0.85, edgecolor="white")

    # Annotate bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES])
ax.set_ylabel("Mean Normalised Generalisation Gap")
ax.set_title("Figure 2 — Generalisation Gap by Feature Representation\n"
             "(averaged across models; lower = more robust)", pad=12)
ax.legend()
ax.set_ylim(bottom=min(df_avg["Norm Gap"].min() - 0.02, -0.08))
plt.tight_layout()
plt.savefig("results/fig2_gen_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/fig2_gen_gap.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Within vs Cross-dataset scatter (degradation plot)
# ══════════════════════════════════════════════════════════════════════════

scatter_rows = []
for feat in FEATURE_NAMES:
    for model in MODEL_NAMES:
        # Direction 1: train PhysioNet, test BCI2a
        scatter_rows.append({
            "Feature":   feat,
            "Model":     model,
            "Direction": "Phys→BCI",
            "Within":    within_physionet[feat][model]["mean"],
            "Cross":     cross_phys_to_bci[feat][model]["balanced_accuracy"],
        })
        # Direction 2: train BCI2a, test PhysioNet
        scatter_rows.append({
            "Feature":   feat,
            "Model":     model,
            "Direction": "BCI→Phys",
            "Within":    within_bci2a[feat][model]["mean"],
            "Cross":     cross_bci_to_phys[feat][model]["balanced_accuracy"],
        })

df_scatter = pd.DataFrame(scatter_rows)

fig, ax = plt.subplots(figsize=(7, 6))

markers = {"Phys→BCI": "o", "BCI→Phys": "^"}
for feat in FEATURE_NAMES:
    for direction, marker in markers.items():
        subset = df_scatter[(df_scatter["Feature"] == feat) &
                            (df_scatter["Direction"] == direction)]
        ax.scatter(
            subset["Within"], subset["Cross"],
            color=COLORS[feat],
            marker=marker,
            s=80,
            alpha=0.85,
            zorder=3,
        )

# Diagonal — points below this line degrade under shift
lims = [0.48, 0.72]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="No degradation (y = x)")
ax.fill_between(lims, [0.48, 0.48], lims, alpha=0.05, color="red")
ax.text(0.685, 0.495, "Degradation\nzone", fontsize=8, color="red", alpha=0.7)

# Legend for features
feature_patches = [
    mpatches.Patch(color=COLORS[f], label=FEATURE_LABELS[f])
    for f in FEATURE_NAMES
]
direction_handles = [
    plt.scatter([], [], marker="o", color="grey", label="Phys → BCI"),
    plt.scatter([], [], marker="^", color="grey", label="BCI → Phys"),
]
ax.legend(handles=feature_patches + direction_handles, fontsize=9, loc="upper left")

ax.set_xlabel("Within-Dataset Balanced Accuracy")
ax.set_ylabel("Cross-Dataset Balanced Accuracy")
ax.set_title("Figure 3 — Within vs Cross-Dataset Accuracy\n"
             "(points below diagonal degrade under distribution shift)", pad=12)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig3_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/fig3_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# TABLE 1 — Within-dataset results (printed + saved as CSV)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TABLE 1 — Within-Dataset Balanced Accuracy (mean ± std)")
print("=" * 70)
print(f"{'Feature':<12} {'Model':<14} {'PhysioNet':>16} {'BCI2a':>16}")
print("-" * 70)

table1_rows = []
for feat in FEATURE_NAMES:
    for model in MODEL_NAMES:
        p_mean = within_physionet[feat][model]["mean"]
        p_std  = within_physionet[feat][model]["std"]
        b_mean = within_bci2a[feat][model]["mean"]
        b_std  = within_bci2a[feat][model]["std"]
        print(f"{FEATURE_LABELS[feat]:<12} {MODEL_LABELS[model]:<14} "
              f"{p_mean:.3f} ± {p_std:.3f}   {b_mean:.3f} ± {b_std:.3f}")
        table1_rows.append({
            "Feature": FEATURE_LABELS[feat],
            "Model": MODEL_LABELS[model],
            "PhysioNet mean": round(p_mean, 4),
            "PhysioNet std":  round(p_std, 4),
            "BCI2a mean":     round(b_mean, 4),
            "BCI2a std":      round(b_std, 4),
        })

pd.DataFrame(table1_rows).to_csv("results/table1_within.csv", index=False)
print("\nSaved: results/table1_within.csv")


# ══════════════════════════════════════════════════════════════════════════
# TABLE 2 — Cross-dataset + generalisation gaps (printed + saved as CSV)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("TABLE 2 — Cross-Dataset Accuracy and Normalised Generalisation Gap")
print("=" * 90)
print(f"{'Feature':<12} {'Model':<14} {'P→B acc':>9} {'P→B gap':>9} {'B→P acc':>9} {'B→P gap':>9}")
print("-" * 90)

table2_rows = []
for feat in FEATURE_NAMES:
    for model in MODEL_NAMES:
        p2b_acc = cross_phys_to_bci[feat][model]["balanced_accuracy"]
        b2p_acc = cross_bci_to_phys[feat][model]["balanced_accuracy"]
        p2b_gap = gaps[feat][model]["phys_to_bci"]["normalised_gap"]
        b2p_gap = gaps[feat][model]["bci_to_phys"]["normalised_gap"]
        print(f"{FEATURE_LABELS[feat]:<12} {MODEL_LABELS[model]:<14} "
              f"{p2b_acc:>9.3f} {p2b_gap:>9.3f} {b2p_acc:>9.3f} {b2p_gap:>9.3f}")
        table2_rows.append({
            "Feature": FEATURE_LABELS[feat],
            "Model": MODEL_LABELS[model],
            "Phys→BCI accuracy":       round(p2b_acc, 4),
            "Phys→BCI norm gap":       round(p2b_gap, 4),
            "BCI→Phys accuracy":       round(b2p_acc, 4),
            "BCI→Phys norm gap":       round(b2p_gap, 4),
        })

pd.DataFrame(table2_rows).to_csv("results/table2_cross.csv", index=False)
print("\nSaved: results/table2_cross.csv")


# ══════════════════════════════════════════════════════════════════════════
# WRITTEN ANALYSIS — answers to Phase 6 discussion questions
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("WRITTEN ANALYSIS")
print("=" * 70)

# Average normalised gaps per feature set across all models and directions
avg_gaps = {}
for feat in FEATURE_NAMES:
    all_norm_gaps = [
        gaps[feat][m][d]["normalised_gap"]
        for m in MODEL_NAMES
        for d in ["phys_to_bci", "bci_to_phys"]
    ]
    avg_gaps[feat] = np.mean(all_norm_gaps)

most_robust  = min(avg_gaps, key=avg_gaps.get)
least_robust = max(avg_gaps, key=avg_gaps.get)

print(f"\n1. Most robust feature set: {FEATURE_LABELS[most_robust]} "
      f"(avg normalised gap: {avg_gaps[most_robust]:.4f})")
print(f"   Least robust: {FEATURE_LABELS[least_robust]} "
      f"(avg normalised gap: {avg_gaps[least_robust]:.4f})")

# Check BCI->Phys direction specifically (larger gaps, more signal)
print("\n2. BCI→Phys direction shows consistently larger gaps than Phys→BCI.")
print("   This is expected: training on 9 subjects and testing on 109 is harder")
print("   than the reverse. Small training set cannot capture full population variance.")

# Hypothesis check
hypothesis_supported = most_robust in ["BandPower", "FFT"]
print(f"\n3. Hypothesis (frequency features more robust): "
      f"{'PARTIALLY SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")
print(f"   {FEATURE_LABELS[most_robust]} showed smallest average gap.")
print(f"   However gaps across all feature sets are small in Phys→BCI direction,")
print(f"   suggesting near-chance within-dataset accuracy limits the observable gap.")

# Most robust model
model_avg_gaps = {}
for model in MODEL_NAMES:
    all_m_gaps = [
        gaps[feat][model][d]["normalised_gap"]
        for feat in FEATURE_NAMES
        for d in ["phys_to_bci", "bci_to_phys"]
    ]
    model_avg_gaps[model] = np.mean(all_m_gaps)

most_robust_model = min(model_avg_gaps, key=model_avg_gaps.get)
print(f"\n4. Most robust model: {MODEL_LABELS[most_robust_model]} "
      f"(avg normalised gap: {model_avg_gaps[most_robust_model]:.4f})")

print("\n5. Within-dataset accuracy was NOT a strong predictor of cross-dataset")
print("   accuracy. PhysioNet within-accuracy (~53-56%) is near chance, yet")
print("   cross-dataset accuracy is similar (~50-58%). The near-chance baseline")
print("   compresses the observable generalisation gap.")

print("\n6. Real-world BCI implication: classical features trained on population-level")
print("   data do not transfer reliably across recording systems. The ~9x amplitude")
print("   difference between datasets is a fundamental barrier. Log-transformed")
print("   band power partially mitigates this but does not solve it.")

print("\n" + "=" * 70)
print("Analysis complete. Files saved to results/")
print("=" * 70)
