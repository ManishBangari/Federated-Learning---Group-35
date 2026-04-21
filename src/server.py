"""
server.py — Flower server with FedAvg strategy.

Handles:
  - Global model aggregation (weighted average of client weights)
  - Per-round metric collection (loss, accuracy)
  - CSV logging via MetricsLogger
  - Plot generation after training completes
"""

import flwr as fl
from flwr.common import Metrics, NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from model import get_model, get_parameters
from utils import MetricsLogger, plot_metrics, set_seed


# ──────────────────────────────────────────────
#  Metric Aggregation Helpers
# ──────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client metrics using weighted average
    (weighted by number of local samples).
    Used for both fit() and evaluate() metric aggregation.
    """
    total_examples = sum(num for num, _ in metrics)

    aggregated = {}
    for num, m in metrics:
        for key, value in m.items():
            if key == "client_id":
                continue
            aggregated[key] = aggregated.get(key, 0.0) + (value * num)

    return {k: v / total_examples for k, v in aggregated.items()}


# ──────────────────────────────────────────────
#  Custom FedAvg Strategy with Logging
# ──────────────────────────────────────────────

class FedAvgWithLogging(FedAvg):
    """
    Extends Flower's built-in FedAvg strategy to:
      - Log aggregated metrics every round
      - Save results to CSV at the end
    """

    def __init__(self, logger: MetricsLogger, experiment_name: str,
                 results_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self._round_counter    = 0
        self._round_start_time = None
        self._last_fit_count   = 0

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate weights. Also records round start time for Cat. 7 timing."""
        import time
        self._round_start_time = time.time()
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        self._round_counter = server_round
        # Number of clients that successfully completed fit this round
        self._last_fit_count = len(results)
        return aggregated_weights, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation metrics + log to CSV.

        Maps client-reported val_accuracy/val_loss -> global_test_accuracy/global_test_loss.
        Also logs Cat. 7 metrics: participated_clients.
        round_completion_time and straggler_ratio are left as None until
        straggler simulation is added (TODO).
        """
        import time
        aggregation_end = time.time()

        loss_aggregated, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Client reports "val_accuracy"; we store as "global_test_accuracy"
        global_acc  = aggregated_metrics.get("val_accuracy", None)
        global_loss = loss_aggregated

        # ── Cat. 7: participated_clients ──
        # Number of clients that returned evaluate results this round
        participated = len(results)

        # ── Cat. 7: round_completion_time ──
        # Wall-clock time from start of fit to end of evaluate aggregation.
        # NOTE: In Flower simulation this is CPU time, not real async time.
        # TODO: Will be replaced with per-client simulated delay once
        #       straggler simulation config is provided.
        round_time = (
            round(aggregation_end - self._round_start_time, 3)
            if self._round_start_time is not None else None
        )

        # ── Cat. 7: straggler_ratio ──
        # TODO: requires straggler simulation — left as None for now.
        straggler_ratio = None

        print(f"[Round {server_round:>3}] "
              f"Global Test Loss: {global_loss:.4f} | "
              f"Global Test Accuracy: {global_acc:.2f}% | "
              f"Participated: {participated}"
              if global_acc is not None else
              f"[Round {server_round:>3}] Global Test Loss: {global_loss:.4f} | "
              f"Participated: {participated}")

        self.logger.log(
            round_num=server_round,
            global_test_accuracy=global_acc,
            global_test_loss=global_loss,
            round_completion_time=round_time,       # Cat. 7 (TODO: straggler sim)
            straggler_ratio=straggler_ratio,        # Cat. 7 (TODO: straggler sim)
            participated_clients=participated,      # Cat. 7
        )

        return loss_aggregated, aggregated_metrics


# ──────────────────────────────────────────────
#  Server Entry Point
# ──────────────────────────────────────────────

def run_server(config: dict):
    """
    Initialize and start the Flower server.

    Args:
        config: Full experiment config dictionary (loaded from YAML).
    """
    set_seed(config["experiment"].get("seed", 42))

    exp_name    = config["experiment"]["name"]
    results_dir = config["logging"].get("results_dir", "results")
    num_rounds  = config["federation"]["num_rounds"]
    num_clients = config["federation"]["num_clients"]
    frac_fit    = config["federation"]["fraction_fit"]
    frac_eval   = config["federation"]["fraction_evaluate"]

    # ── Logger ──
    logger = MetricsLogger(exp_name, results_dir)

    # ── Initial global model parameters ──
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["dataset"]["num_classes"],
    )
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(model)
    )

    # ── Strategy ──
    strategy = FedAvgWithLogging(
        logger=logger,
        experiment_name=exp_name,
        results_dir=results_dir,
        # FedAvg hyperparameters
        fraction_fit=frac_fit,
        fraction_evaluate=frac_eval,
        min_fit_clients=max(1, int(frac_fit * num_clients)),
        min_evaluate_clients=max(1, int(frac_eval * num_clients)),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        # Metric aggregation functions
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # ── Flower Server Config ──
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Algorithm  : FedAvg")
    print(f"  Dataset    : {config['dataset']['name'].upper()}")
    print(f"  Clients    : {num_clients} (frac_fit={frac_fit})")
    print(f"  Rounds     : {num_rounds}")
    print(f"  Partition  : {config['dataset']['partition']}")
    print(f"{'='*60}\n")

    # ── Start Server ──
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
    )

    # ── Save results ──
    logger.save()
    if config["logging"].get("save_results", True):
        plot_metrics(exp_name, results_dir)

    print(f"\n[Done] Experiment '{exp_name}' complete.")