#!/usr/bin/env python3
"""
Clean figure-generation script for the revised MultiCausGRN manuscript.

This script removes obsolete intermediate figures and old "causal/no causal" wording.
It generates publication-ready figures using the audited results reported in the revision.

Expected optional input files:
  - Channel1.csv / Channel2.csv or Channel1 (2).csv / Channel2 (2).csv for PCA plots
  - baseline_log.txt and trrust_log.txt for validation training curves
  - predicted_edges_TRRUST_seed_2.csv for hub-TF analysis

Example:
  python multicausgrn_figures_clean.py --out_dir Figures \
      --tf_embeddings "Channel1 (2).csv" \
      --target_embeddings "Channel2 (2).csv" \
      --baseline_log "internal_prior_seeds.txt" \
      --trrust_log "Trrust_Prior.txt" \
      --pred_edges "predicted_edges_TRRUST_seed_2.csv"

If optional files are not provided, only figures that do not require them are generated.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------
# Audited PBMC results used in the revised manuscript
# ---------------------------------------------------------------------
SEEDS = np.array([1, 2, 4, 8, 16])

# Baseline = leakage-free train-positive adjacency baseline.
BASELINE_AUPRC = np.array([0.448205, 0.686437, 0.386378, 0.763342, 0.667377])
BASELINE_AUROC = np.array([0.397569, 0.703125, 0.320312, 0.703559, 0.645833])

# TRRUST prior = leakage-controlled external TRRUST prior, 36 edges.
TRRUST_AUPRC = np.array([0.658000, 0.781800, 0.739800, 0.761235, 0.773543])
TRRUST_AUROC = np.array([0.658000, 0.712700, 0.656300, 0.680122, 0.703125])

# Seed-2 sensitivity analysis under increasingly imbalanced test sets.
IMBALANCE_RATIOS = ["1:1", "1:5", "1:10", "1:50"]
IMBALANCE_AUROC = np.array([0.713, 0.757, 0.746, 0.742])
IMBALANCE_AUPRC = np.array([0.782, 0.636, 0.474, 0.202])
IMBALANCE_POSITIVES = np.array([32, 32, 32, 32])
IMBALANCE_NEGATIVES = np.array([32, 160, 320, 1600])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_out_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(path: Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


# ---------------------------------------------------------------------
# Figure 1: mean ± SD performance
# ---------------------------------------------------------------------
def plot_performance_mean_std(out_dir: Path) -> None:
    labels = ["Baseline", "TRRUST Prior"]

    auprc_means = [mean_std(BASELINE_AUPRC)[0], mean_std(TRRUST_AUPRC)[0]]
    auprc_stds = [mean_std(BASELINE_AUPRC)[1], mean_std(TRRUST_AUPRC)[1]]

    auroc_means = [mean_std(BASELINE_AUROC)[0], mean_std(TRRUST_AUROC)[0]]
    auroc_stds = [mean_std(BASELINE_AUROC)[1], mean_std(TRRUST_AUROC)[1]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].bar(labels, auprc_means, yerr=auprc_stds, capsize=5)
    axes[0].set_title("Test AUPRC")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, auroc_means, yerr=auroc_stds, capsize=5)
    axes[1].set_title("Test AUROC")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)

    save_figure(out_dir / "Figure_performance_mean_std.png")
    plt.close(fig)

    print("Baseline AUPRC mean±SD:", f"{auprc_means[0]:.3f} ± {auprc_stds[0]:.3f}")
    print("TRRUST AUPRC mean±SD:", f"{auprc_means[1]:.3f} ± {auprc_stds[1]:.3f}")
    print("Baseline AUROC mean±SD:", f"{auroc_means[0]:.3f} ± {auroc_stds[0]:.3f}")
    print("TRRUST AUROC mean±SD:", f"{auroc_means[1]:.3f} ± {auroc_stds[1]:.3f}")


# ---------------------------------------------------------------------
# Figure 2: seed-wise stability boxplots
# ---------------------------------------------------------------------
def plot_seed_stability_boxplot(out_dir: Path) -> None:
    labels = ["Baseline", "TRRUST Prior"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].boxplot([BASELINE_AUPRC, TRRUST_AUPRC], tick_labels=labels, showmeans=True)
    axes[0].set_title("Test AUPRC Across Five Random Seeds")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot([BASELINE_AUROC, TRRUST_AUROC], tick_labels=labels, showmeans=True)
    axes[1].set_title("Test AUROC Across Five Random Seeds")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)

    save_figure(out_dir / "Figure_seed_stability_AUPRC_AUROC.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: sensitivity under class imbalance
# ---------------------------------------------------------------------
def plot_imbalance_sensitivity(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(IMBALANCE_RATIOS, IMBALANCE_AUROC, marker="o", linewidth=2, label="AUROC")
    ax.plot(IMBALANCE_RATIOS, IMBALANCE_AUPRC, marker="s", linewidth=2, label="AUPRC")

    ax.set_xlabel("Positive : Negative Ratio")
    ax.set_ylabel("Performance")
    ax.set_title("Sensitivity Analysis Under Increasing Class Imbalance")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    save_figure(out_dir / "Figure_imbalance_sensitivity.png")
    plt.close(fig)

    table = pd.DataFrame({
        "ratio": IMBALANCE_RATIOS,
        "positives": IMBALANCE_POSITIVES,
        "negatives": IMBALANCE_NEGATIVES,
        "AUROC": IMBALANCE_AUROC,
        "AUPRC": IMBALANCE_AUPRC,
    })
    table_path = out_dir / "Table_imbalance_sensitivity.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


# ---------------------------------------------------------------------
# Figure 4: training curves from log files
# ---------------------------------------------------------------------
def extract_runs_metrics(log_path: str | Path, max_epochs: int = 90) -> Tuple[np.ndarray, np.ndarray]:
    text = Path(log_path).read_text(errors="ignore")
    pattern = r"Epoch:(\d+)\s+train loss:[^\n]*?AUC:([0-9.]+)\s+AUPR:([0-9.]+)"

    epochs, aucs, auprs = [], [], []
    for match in re.finditer(pattern, text):
        epochs.append(int(match.group(1)))
        aucs.append(float(match.group(2)))
        auprs.append(float(match.group(3)))

    runs_auc, runs_aupr = [], []
    current_auc, current_aupr = [], []

    for epoch, auc, aupr in zip(epochs, aucs, auprs):
        if epoch == 1 and current_auc:
            runs_auc.append(current_auc)
            runs_aupr.append(current_aupr)
            current_auc, current_aupr = [], []
        current_auc.append(auc)
        current_aupr.append(aupr)

    if current_auc:
        runs_auc.append(current_auc)
        runs_aupr.append(current_aupr)

    runs_auc = [run[:max_epochs] for run in runs_auc if len(run) >= max_epochs]
    runs_aupr = [run[:max_epochs] for run in runs_aupr if len(run) >= max_epochs]
    return np.asarray(runs_auc), np.asarray(runs_aupr)


def plot_training_curves(out_dir: Path, baseline_log: str | None, trrust_log: str | None) -> None:
    if not baseline_log or not trrust_log:
        print("Skipping training curves: --baseline_log and --trrust_log were not provided.")
        return

    baseline_auc, baseline_aupr = extract_runs_metrics(baseline_log)
    trrust_auc, trrust_aupr = extract_runs_metrics(trrust_log)

    if baseline_auc.size == 0 or trrust_auc.size == 0:
        print("Skipping training curves: no complete 90-epoch runs found in one or both logs.")
        return

    epochs = np.arange(1, baseline_auc.shape[1] + 1)

    def plot_mean_std(ax, runs: np.ndarray, label: str) -> None:
        mean = runs.mean(axis=0)
        std = runs.std(axis=0, ddof=1) if runs.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(epochs, mean, label=label, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_mean_std(axes[0], baseline_aupr, "Baseline")
    plot_mean_std(axes[0], trrust_aupr, "TRRUST Prior")
    axes[0].set_title("Validation AUPR")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUPR")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    plot_mean_std(axes[1], baseline_auc, "Baseline")
    plot_mean_std(axes[1], trrust_auc, "TRRUST Prior")
    axes[1].set_title("Validation AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    save_figure(out_dir / "Figure_training_curves_AUPR_AUROC.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 5: PCA of learned embeddings
# ---------------------------------------------------------------------
def load_embedding(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    # Works for both index-column CSV and first-column gene-name CSV.
    names = df.iloc[:, 0].astype(str).values
    features = df.iloc[:, 1:].values.astype(float)
    return names, features


def run_pca(features: np.ndarray, names: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    x = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    return pd.DataFrame({"Gene": names, "PC1": pcs[:, 0], "PC2": pcs[:, 1]}), pca.explained_variance_ratio_


def remove_extreme_for_plot(df: pd.DataFrame, q: float = 0.97) -> pd.DataFrame:
    df = df.copy()
    df["dist"] = np.sqrt(df["PC1"] ** 2 + df["PC2"] ** 2)
    cutoff = df["dist"].quantile(q)
    return df[df["dist"] <= cutoff].copy()


def add_clusters(df: pd.DataFrame, ax, n_clusters: int = 3, title_prefix: str = "Cluster") -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("No points available for clustering.")

    n_clusters = min(n_clusters, len(df))
    coords = df[["PC1", "PC2"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(coords)

    ax.scatter(df["PC1"], df["PC2"], c=df["Cluster"], s=35, alpha=0.75, edgecolor="black", linewidth=0.3)

    for cluster_id in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cluster_id]
        if len(sub) < 3:
            continue
        center_x, center_y = sub["PC1"].mean(), sub["PC2"].mean()
        width = max(2.2 * sub["PC1"].std(), 0.5)
        height = max(2.2 * sub["PC2"].std(), 0.5)
        ellipse = Ellipse((center_x, center_y), width=width, height=height, fill=False, linestyle="--", linewidth=1.5)
        ax.add_patch(ellipse)
        ax.text(center_x, center_y, f"{title_prefix} {cluster_id + 1}", fontsize=9, fontweight="bold")

    return df


def label_points(ax, df: pd.DataFrame, genes_to_label: Iterable[str]) -> None:
    genes_upper = df["Gene"].str.upper()
    for gene in genes_to_label:
        sub = df[genes_upper == gene.upper()]
        if not sub.empty:
            row = sub.iloc[0]
            ax.annotate(row["Gene"], (row["PC1"], row["PC2"]), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold")


def plot_embedding_pca(out_dir: Path, tf_embeddings: str | None, target_embeddings: str | None) -> None:
    if not tf_embeddings or not target_embeddings:
        print("Skipping PCA plot: --tf_embeddings and --target_embeddings were not provided.")
        return

    tf_names, tf_features = load_embedding(tf_embeddings)
    target_names, target_features = load_embedding(target_embeddings)

    tf_pca, tf_var = run_pca(tf_features, tf_names)
    target_pca, target_var = run_pca(target_features, target_names)

    tf_plot = remove_extreme_for_plot(tf_pca, q=0.97)
    target_plot = remove_extreme_for_plot(target_pca, q=0.97)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    add_clusters(tf_plot, axes[0], n_clusters=2, title_prefix="TF cluster")
    label_points(axes[0], tf_pca, ["GATA3", "TBX21", "SPIB", "MAF"])
    axes[0].set_title("TF Embeddings (Zoomed PCA)")
    axes[0].set_xlabel(f"PC1 ({tf_var[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({tf_var[1] * 100:.1f}%)")
    axes[0].grid(alpha=0.3)

    add_clusters(target_plot, axes[1], n_clusters=3, title_prefix="Target cluster")
    label_points(axes[1], target_pca, ["GATA3", "LAG3", "IL7R", "CCR7"])
    axes[1].set_title("Target Gene Embeddings (Zoomed PCA)")
    axes[1].set_xlabel(f"PC1 ({target_var[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({target_var[1] * 100:.1f}%)")
    axes[1].grid(alpha=0.3)

    save_figure(out_dir / "Figure_PCA_embeddings_clustered_zoomed.png", dpi=600)
    plt.close(fig)

    pca_summary = pd.DataFrame({
        "embedding": ["TF", "Target"],
        "PC1_variance": [tf_var[0], target_var[0]],
        "PC2_variance": [tf_var[1], target_var[1]],
        "PC1_PC2_total": [tf_var[:2].sum(), target_var[:2].sum()],
    })
    pca_summary.to_csv(out_dir / "PCA_variance_summary.csv", index=False)


# ---------------------------------------------------------------------
# Figure 6: hub TFs from predicted edges
# ---------------------------------------------------------------------
def plot_top_hub_tfs(out_dir: Path, pred_edges: str | None, top_n_edges: int = 100, top_n_tfs: int = 10) -> None:
    if not pred_edges:
        print("Skipping hub-TF plot: --pred_edges was not provided.")
        return

    df = pd.read_csv(pred_edges)
    required_cols = {"TF_name", "Score"}
    if not required_cols.issubset(df.columns):
        print(f"Skipping hub-TF plot: prediction file must contain columns {required_cols}.")
        return

    top_edges = df.sort_values("Score", ascending=False).head(top_n_edges)
    hub_tfs = top_edges.groupby("TF_name").size().reset_index(name="Degree").sort_values("Degree", ascending=False)
    top_plot = hub_tfs.head(top_n_tfs).sort_values("Degree", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_plot["TF_name"], top_plot["Degree"])
    ax.set_xlabel("Number of Predicted Target Genes")
    ax.set_ylabel("Transcription Factors")
    ax.set_title(f"Top Hub TFs in the Top {top_n_edges} Predicted Interactions")

    save_figure(out_dir / "Figure_top_hub_TFs_TRRUST_top100.png")
    plt.close(fig)

    hub_tfs.to_csv(out_dir / "Top_Hub_TFs_TRRUST_top100.csv", index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate revised MultiCausGRN manuscript figures.")
    parser.add_argument("--out_dir", type=str, default="Figures", help="Directory where figures and tables will be saved.")
    parser.add_argument("--baseline_log", type=str, default=None, help="Optional log file for baseline training curves.")
    parser.add_argument("--trrust_log", type=str, default=None, help="Optional log file for TRRUST training curves.")
    parser.add_argument("--tf_embeddings", type=str, default=None, help="Optional Channel1/TF embedding CSV path.")
    parser.add_argument("--target_embeddings", type=str, default=None, help="Optional Channel2/target embedding CSV path.")
    parser.add_argument("--pred_edges", type=str, default=None, help="Optional predicted edges CSV for hub-TF plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    plot_performance_mean_std(out_dir)
    plot_seed_stability_boxplot(out_dir)
    plot_imbalance_sensitivity(out_dir)
    plot_training_curves(out_dir, args.baseline_log, args.trrust_log)
    plot_embedding_pca(out_dir, args.tf_embeddings, args.target_embeddings)
    plot_top_hub_tfs(out_dir, args.pred_edges)


if __name__ == "__main__":
    main()
