"""
utils.py — Seeding, logging, CSV saving, and plotting utilities.
"""

import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ──────────────────────────────────────────────
#  Seeding (MANDATORY — fixed seed = 42)
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds fixed to {seed}")


# ──────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    return device


# ──────────────────────────────────────────────
#  Directory helpers
# ──────────────────────────────────────────────

def make_dirs(results_dir: str = "results"):
    Path(f"{results_dir}/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"{results_dir}/plots").mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
#  CSV Logging
# ──────────────────────────────────────────────

class MetricsLogger:
    """
    Logs per-round metrics to a CSV file.
    Columns: round, train_loss, train_accuracy, val_loss, val_accuracy
    """

    def __init__(self, experiment_name: str, results_dir: str = "results"):
        make_dirs(results_dir)
        self.path = os.path.join(results_dir, "metrics", f"{experiment_name}.csv")
        self.records = []
        print(f"[Logger] Metrics will be saved to: {self.path}")

    def log(self, round_num: int, **kwargs):
        """Log a dictionary of metrics for a given round."""
        record = {"round": round_num, **kwargs}
        self.records.append(record)

    def save(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.path, index=False)
        print(f"[Logger] Saved metrics → {self.path}")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# ──────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────

def plot_metrics(
    experiment_name: str,
    results_dir: str = "results",
    show: bool = False
):
    """
    Load saved CSV and produce accuracy + loss curves.
    Saves both .png and .pdf to results/plots/
    """
    csv_path = os.path.join(results_dir, "metrics", f"{experiment_name}.csv")
    if not os.path.exists(csv_path):
        print(f"[Plot] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Experiment: {experiment_name}", fontsize=14, fontweight="bold")

    # ── Accuracy ──
    if "val_accuracy" in df.columns:
        axes[0].plot(df["round"], df["val_accuracy"], label="Val Accuracy", color="steelblue")
    if "train_accuracy" in df.columns:
        axes[0].plot(df["round"], df["train_accuracy"], label="Train Accuracy",
                     color="steelblue", linestyle="--", alpha=0.6)
    axes[0].set_title("Accuracy over Rounds")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()

    # ── Loss ──
    if "val_loss" in df.columns:
        axes[1].plot(df["round"], df["val_loss"], label="Val Loss", color="tomato")
    if "train_loss" in df.columns:
        axes[1].plot(df["round"], df["train_loss"], label="Train Loss",
                     color="tomato", linestyle="--", alpha=0.6)
    axes[1].set_title("Loss over Rounds")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()

    base_path = os.path.join(results_dir, "plots", experiment_name)
    plt.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
    print(f"[Plot] Saved → {base_path}.png / .pdf")

    if show:
        plt.show()
    plt.close()