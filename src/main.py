"""
main.py — Unified entry point for FL experiments using Flower simulation.

Usage:
    python main.py --config configs/mnist_fedavg_10clients_iid.yaml

Flower's `start_simulation()` is used so that both server and all clients
run in a single process — no need to launch separate terminals.
"""

import argparse
import yaml
import torch
import flwr as fl
from flwr.common import NDArrays
from typing import Dict

# ── Project imports ──
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, get_device, make_dirs, MetricsLogger, plot_metrics
from src.data import load_dataset, partition_data
from src.model import get_model, get_parameters
from src.client import FedAvgClient, set_parameters
from src.server import FedAvgWithLogging, weighted_average


# ──────────────────────────────────────────────
#  Config Loader
# ──────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ──────────────────────────────────────────────
#  Client Factory (called by Flower simulation)
# ──────────────────────────────────────────────

def make_client_fn(
    train_dataset,
    test_dataset,
    client_indices,
    config: dict,
    device: torch.device,
):
    """
    Returns a client_fn compatible with Flower's simulation API.
    Flower calls client_fn(cid) for each sampled client per round.
    """
    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        return FedAvgClient(
            client_id=cid_int,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            client_indices=client_indices[cid_int],
            config=config,
            device=device,
        )
    return client_fn


# ──────────────────────────────────────────────
#  Main Simulation
# ──────────────────────────────────────────────

def run_simulation(config: dict):
    # 1. Seed everything
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    # 2. Device
    device = get_device()

    # 3. Output dirs
    results_dir = config["logging"].get("results_dir", "results")
    make_dirs(results_dir)

    # 4. Load datasets
    dataset_name = config["dataset"]["name"]
    train_dataset, test_dataset = load_dataset(dataset_name)

    # 5. Partition training data across clients
    num_clients   = config["federation"]["num_clients"]
    partition     = config["dataset"]["partition"]
    alpha         = config["dataset"].get("dirichlet_alpha", 0.5) or 0.5

    client_indices = partition_data(
        dataset=train_dataset,
        num_clients=num_clients,
        partition=partition,
        dirichlet_alpha=alpha,
        seed=seed,
    )

    # 6. Logger -- suffix experiment name with partition/alpha so results
    #    from different alpha values never overwrite each other
    exp_name = config["experiment"]["name"]
    if partition == "noniid_dirichlet":
        exp_name = f"{exp_name}_alpha{str(alpha).replace('.', '')}"
    else:
        exp_name = f"{exp_name}_iid"

    # Compute model size in MB (params x 4 bytes for float32)
    _tmp_model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["dataset"]["num_classes"],
    )
    _num_params       = sum(p.numel() for p in _tmp_model.parameters())
    model_size_mb     = (_num_params * 4) / (1024 ** 2)   # float32 = 4 bytes
    clients_per_round = max(1, int(config["federation"]["fraction_fit"] * num_clients))

    logger = MetricsLogger(
        experiment_name=exp_name,
        results_dir=results_dir,
        convergence_threshold=80.0,
        model_size_mb=model_size_mb,
        num_clients_per_round=clients_per_round,
    )

    # 7. Initial global model
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["dataset"]["num_classes"],
    )
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(model)
    )

    # 8. Strategy
    frac_fit  = config["federation"]["fraction_fit"]
    frac_eval = config["federation"]["fraction_evaluate"]

    strategy = FedAvgWithLogging(
        logger=logger,
        experiment_name=exp_name,
        results_dir=results_dir,
        fraction_fit=frac_fit,
        fraction_evaluate=frac_eval,
        min_fit_clients=max(1, int(frac_fit * num_clients)),
        min_evaluate_clients=max(1, int(frac_eval * num_clients)),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 9. Client factory
    client_fn = make_client_fn(
        train_dataset, test_dataset, client_indices, config, device
    )

    # 10. Print experiment summary
    print(f"\n{'='*60}")
    print(f"  Experiment  : {exp_name}")
    print(f"  Algorithm   : {config['experiment']['algorithm'].upper()}")
    print(f"  Dataset     : {dataset_name.upper()}")
    print(f"  Partition   : {partition}" +
          (f" (α={alpha})" if partition == "noniid_dirichlet" else ""))
    print(f"  Clients     : {num_clients}  |  Frac fit: {frac_fit}")
    print(f"  Rounds      : {config['federation']['num_rounds']}")
    print(f"  Local Epochs: {config['federation']['local_epochs']}")
    print(f"  Batch Size  : {config['federation']['batch_size']}")
    print(f"  LR          : {config['optimizer']['lr']}  |  "
          f"Momentum: {config['optimizer']['momentum']}")
    print(f"{'='*60}\n")

    # 11. Launch Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(
            num_rounds=config["federation"]["num_rounds"]
        ),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # 12. Save metrics + plots
    logger.save()
    if config["logging"].get("save_results", True):
        plot_metrics(exp_name, results_dir)

    print(f"\n[Done] '{exp_name}' — results saved to {results_dir}/")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Learning Experiments — Flower Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/mnist_fedavg_10clients_iid.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_simulation(config)