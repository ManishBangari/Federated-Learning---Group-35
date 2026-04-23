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

    Columns tracked every round:
      - round
      - global_test_accuracy   : accuracy on held-out global test set (%)
      - global_test_loss       : cross-entropy loss on global test set
      - convergence_round      : first round accuracy exceeded threshold (-1 if not reached)
      - comm_cost_mb           : cumulative MB transmitted up to this round
      [Cat. 7 – System Heterogeneity]
      - round_completion_time  : wall-clock seconds for this round (TODO: fill when straggler sim added)
      - straggler_ratio        : fraction of sampled clients that were stragglers (TODO: fill when straggler sim added)
      - participated_clients   : number of clients that actually completed this round

    Also prints a summary at the end.
    """

    def __init__(
        self,
        experiment_name: str,
        results_dir: str = "results",
        convergence_threshold: float = 80.0,   # % accuracy
        model_size_mb: float = 0.0,
        num_clients_per_round: int = 1,
    ):
        make_dirs(results_dir)
        self.path = os.path.join(results_dir, "metrics", f"{experiment_name}.csv")
        self.records = []
        self.convergence_threshold  = convergence_threshold
        self.convergence_round      = -1        # -1 = not yet reached
        self.model_size_mb          = model_size_mb
        self.num_clients_per_round  = num_clients_per_round
        self._cumulative_comm_mb    = 0.0
        self._round_start_time      = None   # set by server before each round
        print(f"[Logger] Metrics will be saved to: {self.path}")
        print(f"[Logger] Convergence threshold : {convergence_threshold}% accuracy")

    def set_model_size(self, model_size_mb: float):
        """Call once after the model is initialised to enable comm cost tracking."""
        self.model_size_mb = model_size_mb
        print(f"[Logger] Model size : {model_size_mb:.4f} MB")

    def log(self, round_num: int, **kwargs):
        """
        Log metrics for one round.

        Expected kwargs:
            global_test_accuracy (float) -- % accuracy on global test set
            global_test_loss     (float) -- cross-entropy loss on global test set
            round_completion_time (float) -- wall-clock seconds for this round [Cat. 7]
            straggler_ratio       (float) -- fraction of stragglers this round  [Cat. 7 - TODO]
            participated_clients  (int)   -- clients that completed this round  [Cat. 7]
        """
        acc  = kwargs.get("global_test_accuracy", None)
        loss = kwargs.get("global_test_loss",     None)

        # ── Convergence round (first time acc >= threshold) ──
        if self.convergence_round == -1 and acc is not None:
            if acc >= self.convergence_threshold:
                self.convergence_round = round_num
                print(f"[Logger] *** Convergence reached at round {round_num} "
                      f"(acc={acc:.2f}% >= {self.convergence_threshold}%) ***")

        # ── Cumulative communication cost ──
        # Each round: server sends model to K clients, clients send updated model back
        # Total per round = 2 x model_size x clients_per_round (upload + download)
        round_comm_mb = 2.0 * self.model_size_mb * self.num_clients_per_round
        self._cumulative_comm_mb += round_comm_mb

        # ── Cat. 7: System Heterogeneity metrics ──
        # round_completion_time: wall-clock time this round took (seconds)
        #   TODO: will be populated once straggler simulation is added
        round_time = kwargs.get("round_completion_time", None)

        # straggler_ratio: fraction of sampled clients that were slow/dropped
        #   TODO: will be populated once straggler simulation is added
        straggler_ratio = kwargs.get("straggler_ratio", None)

        # participated_clients: how many clients actually returned results this round
        #   (already available from Flower results — passed in from server)
        participated = kwargs.get("participated_clients", None)

        record = {
            "round":                  round_num,
            "global_test_accuracy":   acc,
            "global_test_loss":       loss,
            "convergence_round":      self.convergence_round,
            "comm_cost_mb":           round(self._cumulative_comm_mb, 4),
            # ── Cat. 7 ──
            "round_completion_time":  round_time,       # TODO: straggler sim
            "straggler_ratio":        straggler_ratio,  # TODO: straggler sim
            "participated_clients":   participated,
        }
        self.records.append(record)

    def save(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.path, index=False)
        print(f"\n[Logger] Saved metrics -> {self.path}")
        print(f"[Logger] Convergence round    : "
              f"{self.convergence_round if self.convergence_round != -1 else 'Not reached'}")
        print(f"[Logger] Total comm cost      : {self._cumulative_comm_mb:.2f} MB")
        # Cat. 7 summary (only printed if data was collected)
        df = pd.DataFrame(self.records)
        if "participated_clients" in df.columns and df["participated_clients"].notna().any():
            avg_part = df["participated_clients"].mean()
            print(f"[Logger] Avg participated/round: {avg_part:.1f} clients")
        if "round_completion_time" in df.columns and df["round_completion_time"].notna().any():
            avg_time = df["round_completion_time"].mean()
            print(f"[Logger] Avg round time        : {avg_time:.2f}s  [TODO: straggler sim]")
        if "straggler_ratio" in df.columns and df["straggler_ratio"].notna().any():
            avg_str = df["straggler_ratio"].mean()
            print(f"[Logger] Avg straggler ratio   : {avg_str:.3f}  [TODO: straggler sim]")

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
    Load saved CSV and produce 3 subplots:
      1. Global Test Accuracy  (with convergence round marker)
      2. Global Test Loss
      3. Cumulative Communication Cost (MB)

    Saves .png to results/plots/
    """
    csv_path = os.path.join(results_dir, "metrics", f"{experiment_name}.csv")
    if not os.path.exists(csv_path):
        print(f"[Plot] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Experiment: {experiment_name}", fontsize=13, fontweight="bold")

    # ── 1. Global Test Accuracy ──
    ax = axes[0]
    if "global_test_accuracy" in df.columns:
        ax.plot(df["round"], df["global_test_accuracy"],
                label="Global Test Accuracy", color="steelblue", linewidth=2)

        # Mark convergence round
        conv_rows = df[df["convergence_round"] > 0]
        if not conv_rows.empty:
            conv_round = int(conv_rows["convergence_round"].iloc[0])
            conv_acc   = df.loc[df["round"] == conv_round, "global_test_accuracy"]
            if not conv_acc.empty:
                ax.axvline(x=conv_round, color="green", linestyle="--",
                           linewidth=1.5, label=f"Convergence @ R{conv_round}")
                ax.scatter([conv_round], [conv_acc.values[0]],
                           color="green", zorder=5, s=60)
    ax.set_title("Global Test Accuracy")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8)

    # ── 2. Global Test Loss ──
    ax = axes[1]
    if "global_test_loss" in df.columns:
        ax.plot(df["round"], df["global_test_loss"],
                label="Global Test Loss", color="tomato", linewidth=2)
    ax.set_title("Global Test Loss")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=8)

    # ── 3. Communication Cost ──
    ax = axes[2]
    if "comm_cost_mb" in df.columns:
        ax.plot(df["round"], df["comm_cost_mb"],
                label="Cumulative Comm Cost", color="darkorange", linewidth=2)
        # Annotate final value
        final_mb = df["comm_cost_mb"].iloc[-1]
        ax.annotate(f"{final_mb:.1f} MB",
                    xy=(df["round"].iloc[-1], final_mb),
                    xytext=(-40, -15), textcoords="offset points",
                    fontsize=8, color="darkorange")
    ax.set_title("Communication Cost (Cumulative)")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Total MB Transmitted")
    ax.legend(fontsize=8)

    plt.tight_layout()

    base_path = os.path.join(results_dir, "plots", experiment_name)
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved -> {base_path}.png")

    if show:
        plt.show()
    plt.close()

# ══════════════════════════════════════════════════════════════════════
#  MANDATORY FIGURES  (Section 6.1)
#  All plots: 300 DPI, PNG, font >= 11pt, numbered captions.
#
#  Figure 1 — Global Accuracy vs Rounds        (all experiments, one plot)
#  Figure 2 — Global Loss vs Rounds            (all experiments, one plot)
#  Figure 3 — Cat. 7 System Heterogeneity      (round time / straggler / partial participation)
#  Figure 4 — IID vs Non-IID Comparison
#  Figure 5 — FedAvg vs Proposed Method        (side-by-side bars or lines)
#
#  MANDATORY TABLE (Section 6.2)
#  generate_results_table() — one row per experiment, best result bolded.
# ══════════════════════════════════════════════════════════════════════

# ── Shared style constants ────────────────────
FONT_SIZE   = 12          # minimum 11pt per spec
TITLE_SIZE  = 13
DPI         = 300
FIGSIZE_WIDE = (16, 6)
FIGSIZE_SQ   = (10, 7)
PALETTE      = sns.color_palette("tab10")


def _apply_style():
    """Apply consistent style to all mandatory figures."""
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "font.size":        FONT_SIZE,
        "axes.titlesize":   TITLE_SIZE,
        "axes.labelsize":   FONT_SIZE,
        "xtick.labelsize":  FONT_SIZE,
        "ytick.labelsize":  FONT_SIZE,
        "legend.fontsize":  FONT_SIZE - 1,
        "figure.titlesize": TITLE_SIZE + 1,
    })


def _save_fig(fig, base_path: str, caption: str):
    """Save figure as PNG (300 DPI), then print caption."""
    fig.savefig(f"{base_path}.png", dpi=DPI, bbox_inches="tight")
    print(f"[Plot] Saved -> {base_path}.png")
    print(f"       Caption: {caption}")
    plt.close(fig)


def _load_all_csvs(results_dir: str) -> dict:
    """
    Load all experiment CSVs from results/metrics/.
    Returns {experiment_name: DataFrame}.
    """
    metrics_dir = os.path.join(results_dir, "metrics")
    data = {}
    if not os.path.exists(metrics_dir):
        print(f"[Plot] Metrics dir not found: {metrics_dir}")
        return data
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.endswith(".csv"):
            continue
        if fname == "results_table.csv":
            continue
        name = fname.replace(".csv", "")
        df   = pd.read_csv(os.path.join(metrics_dir, fname))
        data[name] = df
    print(f"[Plot] Loaded {len(data)} experiment CSV(s).")
    return data


# ──────────────────────────────────────────────
#  Figure 1 — Global Accuracy vs Rounds
#  All experiments on one plot.
# ──────────────────────────────────────────────

def plot_figure1_accuracy(results_dir: str = "results", show: bool = False):
    """
    Figure 1: Global Test Accuracy vs Communication Rounds.
    One line per experiment, all on a single plot.
    """
    _apply_style()
    data = _load_all_csvs(results_dir)
    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for i, (name, df) in enumerate(data.items()):
        if "global_test_accuracy" not in df.columns:
            continue
        ax.plot(
            df["round"], df["global_test_accuracy"],
            label=name, color=PALETTE[i % len(PALETTE)], linewidth=1.8
        )

    ax.set_title("Figure 1: Global Test Accuracy vs Communication Rounds", fontsize=TITLE_SIZE)
    ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=FONT_SIZE)
    ax.legend(loc="lower right", fontsize=FONT_SIZE - 2, ncol=2)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    base  = os.path.join(results_dir, "plots", "figure1_accuracy_vs_rounds")
    cap   = ("Figure 1: Global test accuracy (%) vs. communication rounds "
             "for all experimental configurations.")
    _save_fig(fig, base, cap)
    if show: plt.show()


# ──────────────────────────────────────────────
#  Figure 2 — Global Loss vs Rounds
#  All experiments on one plot.
# ──────────────────────────────────────────────

def plot_figure2_loss(results_dir: str = "results", show: bool = False):
    """
    Figure 2: Global Test Loss vs Communication Rounds.
    One line per experiment, all on a single plot.
    """
    _apply_style()
    data = _load_all_csvs(results_dir)
    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for i, (name, df) in enumerate(data.items()):
        if "global_test_loss" not in df.columns:
            continue
        ax.plot(
            df["round"], df["global_test_loss"],
            label=name, color=PALETTE[i % len(PALETTE)], linewidth=1.8
        )

    ax.set_title("Figure 2: Global Test Loss vs Communication Rounds", fontsize=TITLE_SIZE)
    ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=FONT_SIZE)
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 2, ncol=2)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    base = os.path.join(results_dir, "plots", "figure2_loss_vs_rounds")
    cap  = ("Figure 2: Global cross-entropy test loss vs. communication rounds "
            "for all experimental configurations.")
    _save_fig(fig, base, cap)
    if show: plt.show()


# ──────────────────────────────────────────────
#  Figure 3 — Cat. 7 System Heterogeneity Metrics
#  round_completion_time, straggler_ratio, participated_clients
# ──────────────────────────────────────────────

def plot_figure3_system_heterogeneity(results_dir: str = "results", show: bool = False):
    """
    Figure 3: Category 7 — System Heterogeneity metrics vs rounds.
      - Subplot A: Round Completion Time (seconds)       [TODO: straggler sim]
      - Subplot B: Straggler Ratio                       [TODO: straggler sim]
      - Subplot C: Participated Clients per Round        [available now]

    Subplots A and B will render as empty with a TODO label until
    straggler simulation is configured.
    """
    _apply_style()
    data = _load_all_csvs(results_dir)
    if not data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Figure 3: Cat. 7 – System Heterogeneity Metrics vs Communication Rounds",
        fontsize=TITLE_SIZE
    )

    for i, (name, df) in enumerate(data.items()):
        c = PALETTE[i % len(PALETTE)]

        # ── A: Round completion time ──
        ax = axes[0]
        if "round_completion_time" in df.columns and df["round_completion_time"].notna().any():
            ax.plot(df["round"], df["round_completion_time"],
                    label=name, color=c, linewidth=1.6)
        ax.set_title("(A) Round Completion Time\n[TODO: straggler sim]", fontsize=FONT_SIZE)
        ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
        ax.set_ylabel("Time (seconds)", fontsize=FONT_SIZE)

        # ── B: Straggler ratio ──
        ax = axes[1]
        if "straggler_ratio" in df.columns and df["straggler_ratio"].notna().any():
            ax.plot(df["round"], df["straggler_ratio"],
                    label=name, color=c, linewidth=1.6)
        ax.set_title("(B) Straggler Ratio\n[TODO: straggler sim]", fontsize=FONT_SIZE)
        ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
        ax.set_ylabel("Straggler Ratio", fontsize=FONT_SIZE)

        # ── C: Participated clients (available now) ──
        ax = axes[2]
        if "participated_clients" in df.columns and df["participated_clients"].notna().any():
            ax.plot(df["round"], df["participated_clients"],
                    label=name, color=c, linewidth=1.6)
        ax.set_title("(C) Accuracy Under Partial Participation\n(Participated Clients/Round)",
                     fontsize=FONT_SIZE)
        ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
        ax.set_ylabel("Clients Participated", fontsize=FONT_SIZE)

    # Add TODO watermark on empty subplots
    for ax_idx, todo_label in [(0, "Awaiting straggler sim config"),
                               (1, "Awaiting straggler sim config")]:
        ax = axes[ax_idx]
        if not ax.lines:
            ax.text(0.5, 0.5, f"TODO\n{todo_label}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=FONT_SIZE, color="gray", style="italic",
                    bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

    for ax in axes:
        ax.legend(fontsize=FONT_SIZE - 2, ncol=1)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()

    base = os.path.join(results_dir, "plots", "figure3_system_heterogeneity")
    cap  = ("Figure 3: Category 7 system heterogeneity metrics. "
            "(A) Round completion time in seconds. "
            "(B) Straggler ratio (fraction of slow/dropped clients). "
            "(C) Number of clients participating per round "
            "(proxy for accuracy under partial participation).")
    _save_fig(fig, base, cap)
    if show: plt.show()


# ──────────────────────────────────────────────
#  Figure 4 — IID vs Non-IID Comparison
# ──────────────────────────────────────────────

def plot_figure4_iid_vs_noniid(results_dir: str = "results", show: bool = False):
    """
    Figure 4: IID vs Non-IID accuracy comparison.
    Groups experiments by dataset + client count, overlays all alpha values.
    Produces one subplot per (dataset, num_clients) combo found in results.
    """
    _apply_style()
    data = _load_all_csvs(results_dir)
    if not data:
        return

    # Group by dataset + client count
    groups = {}   # key: "dataset_Nclients"  value: {label: df}
    for name, df in data.items():
        parts = name.split("_")
        # Expected pattern: fedavg_<dataset>_<N>clients_<partition_tag>
        try:
            algo    = parts[0]
            dataset = parts[1]
            clients = parts[2]                        # e.g. "10clients"
            ptag    = "_".join(parts[3:])             # e.g. "iid" or "alpha05"
            gkey    = f"{dataset}_{clients}"
            groups.setdefault(gkey, {})[ptag] = df
        except IndexError:
            continue

    if not groups:
        print("[Plot] No experiments found for Figure 4.")
        return

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6), squeeze=False)
    fig.suptitle("Figure 4: IID vs Non-IID Global Test Accuracy Comparison",
                 fontsize=TITLE_SIZE)

    # Sorted so IID comes last (visual reference line)
    partition_order = ["alpha001", "alpha01", "alpha05", "alpha10", "iid"]

    for col, (gkey, exp_dict) in enumerate(sorted(groups.items())):
        ax = axes[0][col]
        for j, ptag in enumerate(partition_order):
            if ptag not in exp_dict:
                continue
            df    = exp_dict[ptag]
            label = "IID" if ptag == "iid" else f"Non-IID α={ptag.replace('alpha','0.')}"
            ls    = "-" if ptag == "iid" else "--"
            if "global_test_accuracy" in df.columns:
                ax.plot(df["round"], df["global_test_accuracy"],
                        label=label, linestyle=ls,
                        color=PALETTE[j % len(PALETTE)], linewidth=1.8)

        ax.set_title(f"{gkey.replace('_', ' | ')}", fontsize=FONT_SIZE)
        ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=FONT_SIZE)
        ax.legend(fontsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()

    base = os.path.join(results_dir, "plots", "figure4_iid_vs_noniid")
    cap  = ("Figure 4: Comparison of global test accuracy under IID and Non-IID "
            "(Dirichlet α ∈ {0.01, 0.1, 0.5, 1.0}) data partitioning strategies "
            "across datasets and client counts.")
    _save_fig(fig, base, cap)
    if show: plt.show()


# ──────────────────────────────────────────────
#  Figure 5 — FedAvg vs Proposed Method
# ──────────────────────────────────────────────

def plot_figure5_fedavg_vs_proposed(
    results_dir: str = "results",
    proposed_method: str = None,     # e.g. "fedprox" — fill once papers are known
    show: bool = False
):
    """
    Figure 5: FedAvg baseline vs proposed method.
    Side-by-side line plots — one for accuracy, one for loss.

    Args:
        proposed_method: algorithm tag to filter (e.g. "fedprox").
                         Leave None until papers are confirmed — will
                         print a TODO and skip gracefully.

    TODO: Populate proposed_method once the 2 FL papers are confirmed.
    """
    _apply_style()

    if proposed_method is None:
        print("[Plot] Figure 5: proposed_method not set — skipping. "
              "Set proposed_method=<algo_name> once papers are confirmed.")
        # Create a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5,
                "Figure 5: FedAvg vs Proposed Method\n\nTODO: Set proposed_method "
                "once FL papers are confirmed.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE + 1, color="gray", style="italic",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))
        ax.axis("off")
        base = os.path.join(results_dir, "plots", "figure5_fedavg_vs_proposed")
        _save_fig(fig, base,
                  "Figure 5: FedAvg baseline vs proposed method (placeholder — TODO).")
        if show: plt.show()
        return

    data = _load_all_csvs(results_dir)
    fedavg_exps   = {k: v for k, v in data.items() if "fedavg"        in k}
    proposed_exps = {k: v for k, v in data.items() if proposed_method in k}

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle(f"Figure 5: FedAvg vs {proposed_method.upper()}",
                 fontsize=TITLE_SIZE)

    for i, (name, df) in enumerate(fedavg_exps.items()):
        c = PALETTE[i % len(PALETTE)]
        if "global_test_accuracy" in df.columns:
            axes[0].plot(df["round"], df["global_test_accuracy"],
                         label=f"FedAvg: {name}", color=c,
                         linestyle="--", linewidth=1.6)
        if "global_test_loss" in df.columns:
            axes[1].plot(df["round"], df["global_test_loss"],
                         label=f"FedAvg: {name}", color=c,
                         linestyle="--", linewidth=1.6)

    for i, (name, df) in enumerate(proposed_exps.items()):
        c = PALETTE[(i + len(fedavg_exps)) % len(PALETTE)]
        if "global_test_accuracy" in df.columns:
            axes[0].plot(df["round"], df["global_test_accuracy"],
                         label=f"{proposed_method.upper()}: {name}", color=c,
                         linestyle="-", linewidth=1.8)
        if "global_test_loss" in df.columns:
            axes[1].plot(df["round"], df["global_test_loss"],
                         label=f"{proposed_method.upper()}: {name}", color=c,
                         linestyle="-", linewidth=1.8)

    axes[0].set_title("Global Test Accuracy", fontsize=FONT_SIZE)
    axes[0].set_xlabel("Communication Round", fontsize=FONT_SIZE)
    axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
    axes[0].legend(fontsize=FONT_SIZE - 3, ncol=2)
    axes[0].grid(True, alpha=0.4)

    axes[1].set_title("Global Test Loss", fontsize=FONT_SIZE)
    axes[1].set_xlabel("Communication Round", fontsize=FONT_SIZE)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=FONT_SIZE)
    axes[1].legend(fontsize=FONT_SIZE - 3, ncol=2)
    axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    base = os.path.join(results_dir, "plots", "figure5_fedavg_vs_proposed")
    cap  = (f"Figure 5: Side-by-side comparison of FedAvg (dashed) vs "
            f"{proposed_method.upper()} (solid) on global test accuracy and loss "
            f"across communication rounds.")
    _save_fig(fig, base, cap)
    if show: plt.show()


# ──────────────────────────────────────────────
#  Mandatory Results Table (Section 6.2)
# ──────────────────────────────────────────────

def generate_results_table(results_dir: str = "results") -> pd.DataFrame:
    """
    Section 6.2: Generate the mandatory summary results table.

    Columns:
      Method | Dataset | #Clients | #Rounds | Test Accuracy (%) |
      Convergence Round | Comm. Cost (MB) | Category Metric

    Best result (highest Test Accuracy) is marked with ** in the table.
    Also saves as CSV to results/metrics/results_table.csv.
    Prints a formatted text table to stdout.
    """
    data = _load_all_csvs(results_dir)
    if not data:
        return pd.DataFrame()

    def parse_experiment_name(name: str):
        tokens = name.split("_")
        if tokens and tokens[0].lower() == "test":
            tokens = tokens[1:]

        algo_map = {
            "fedavg": "FedAvg",
            "fedasync": "FedAsync",
            "fedcs": "FedCS",
        }
        algo_token = tokens[0].lower() if tokens else "?"
        algo = algo_map.get(algo_token, algo_token.upper())
        dataset = tokens[1].upper() if len(tokens) > 1 else "?"

        n_clients = "?"
        if len(tokens) > 2:
            client_token = tokens[2]
            if client_token.endswith("clients"):
                try:
                    n_clients = int(client_token.replace("clients", ""))
                except ValueError:
                    n_clients = "?"
            else:
                digits = "".join(ch for ch in client_token if ch.isdigit())
                n_clients = int(digits) if digits else "?"

        partition = "_".join(tokens[3:]) if len(tokens) > 3 else "?"
        return algo, dataset, n_clients, partition

    rows = []
    for name, df in data.items():
        if df.empty:
            continue

        algo, dataset, n_clients, partition = parse_experiment_name(name)
        n_rounds   = int(df["round"].max()) if "round" in df.columns else "?"
        best_acc   = round(df["global_test_accuracy"].max(), 2) \
                     if "global_test_accuracy" in df.columns else "?"
        conv_round = df["convergence_round"].replace(-1, np.nan).dropna()
        conv_r     = int(conv_round.iloc[0]) if not conv_round.empty else "N/A"
        comm_mb    = round(df["comm_cost_mb"].iloc[-1], 2) \
                     if "comm_cost_mb" in df.columns else "?"

        cat7 = "?"
        if "participated_clients" in df.columns and df["participated_clients"].notna().any():
            cat7 = f"Avg part.={df['participated_clients'].mean():.1f}"

        rows.append({
            "Method":             algo,
            "Dataset":            dataset,
            "#Clients":           n_clients,
            "Partition":          partition,
            "#Rounds":            n_rounds,
            "Test Accuracy (%)":  best_acc,
            "Convergence Round":  conv_r,
            "Comm. Cost (MB)":    comm_mb,
            "Category Metric":    cat7,
            "_exp_name":          name,
        })

    if not rows:
        print("[Table] No results to tabulate yet.")
        return pd.DataFrame()

    table_df = pd.DataFrame(rows)
    table_df["Test Accuracy (%)"] = table_df["Test Accuracy (%)"].astype(object)

    # ── Bold (mark) best accuracy ──
    numeric_accuracy = pd.to_numeric(table_df["Test Accuracy (%)"], errors="coerce")
    if numeric_accuracy.notna().any():
        best_idx = numeric_accuracy.idxmax()
        best_val = numeric_accuracy.loc[best_idx]
        table_df.loc[best_idx, "Test Accuracy (%)"] = f"**{best_val:.2f}**"

    # Drop internal column before saving
    save_df = table_df.drop(columns=["_exp_name"])
    out_path = os.path.join(results_dir, "metrics", "results_table.csv")
    save_df.to_csv(out_path, index=False)
    print(f"\n[Table] Results table saved -> {out_path}")

    # Print formatted table to stdout
    print("\n" + "=" * 100)
    print("  SECTION 6.2 — RESULTS SUMMARY TABLE")
    print("=" * 100)
    print(save_df.to_string(index=False))
    print("=" * 100)
    print("  ** = best result\n")

    return save_df


# ──────────────────────────────────────────────
#  Convenience: generate ALL mandatory figures
# ──────────────────────────────────────────────

def generate_all_figures(
    results_dir: str = "results",
    proposed_method: str = None,
    show: bool = False
):
    """
    Generate all 5 mandatory figures + results table in one call.
    Call this after all experiments have completed.

    Usage:
        from src.utils import generate_all_figures
        generate_all_figures(results_dir="results", proposed_method="fedprox")
    """
    print("\n[Figures] Generating all mandatory figures...")
    plot_figure1_accuracy(results_dir, show)
    plot_figure2_loss(results_dir, show)
    plot_figure3_system_heterogeneity(results_dir, show)
    plot_figure4_iid_vs_noniid(results_dir, show)
    plot_figure5_fedavg_vs_proposed(results_dir, proposed_method, show)
    generate_results_table(results_dir)
    print("[Figures] All done. Check results/plots/ and results/metrics/results_table.csv")