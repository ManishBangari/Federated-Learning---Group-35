"""
main_no_ray.py — FL simulation supporting FedAvg, FedAsync, FedCS.
No Ray required. Cross-platform (Linux + Windows).

Usage:
    python main_no_ray.py --config configs/mnist_fedavg_10clients_iid.yaml
    python main_no_ray.py --config configs/test_fedasync.yaml
    python main_no_ray.py --config configs/test_fedcs.yaml
"""

import argparse
import yaml
import random
import time
import numpy as np
import torch
from collections import OrderedDict
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, get_device, make_dirs, MetricsLogger, plot_metrics
from src.data import load_dataset, partition_data
from src.model import get_model, get_parameters
from src.client import FedAvgClient
from src.fedasync import FedAsyncClient, run_fedasync
from src.fedcs import FedCSClient, run_fedcs


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fedavg_aggregate(results):
    total_samples = sum(n for n, _ in results)
    aggregated    = [np.zeros_like(w) for w in results[0][1]]
    for num_samples, weights in results:
        for i, w in enumerate(weights):
            aggregated[i] += w * (num_samples / total_samples)
    return aggregated


def run_fedavg(config, clients, global_weights, logger, num_rounds, seed):
    num_clients       = len(clients)
    clients_per_round = max(1, int(config["federation"]["fraction_fit"] * num_clients))
    rng               = random.Random(seed)
    try:
        for rnd in range(1, num_rounds + 1):
            round_start = time.time()
            sampled_ids = rng.sample(range(num_clients), clients_per_round)
            fit_results = []
            for cid in sampled_ids:
                weights, n_samples, _ = clients[cid].fit(parameters=global_weights, config={})
                fit_results.append((n_samples, weights))
            global_weights = fedavg_aggregate(fit_results)
            eval_losses, eval_accs, eval_counts = [], [], []
            for cid in sampled_ids:
                loss, n, metrics = clients[cid].evaluate(parameters=global_weights, config={})
                eval_losses.append(loss * n)
                eval_accs.append(metrics["val_accuracy"] * n)
                eval_counts.append(n)
            total       = sum(eval_counts)
            global_loss = sum(eval_losses) / total
            global_acc  = sum(eval_accs)   / total
            round_time  = round(time.time() - round_start, 3)
            print(f"[FedAvg Round {rnd:>3}/{num_rounds}]  "
                  f"Loss: {global_loss:.4f}  |  Accuracy: {global_acc:.2f}%  |  "
                  f"Time: {round_time:.1f}s  |  Clients: {len(sampled_ids)}")
            logger.log(round_num=rnd, global_test_accuracy=global_acc,
                       global_test_loss=global_loss, round_completion_time=round_time,
                       straggler_ratio=None, participated_clients=len(sampled_ids))
            logger.save()
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Saving {len(logger.records)} completed rounds...")
    return global_weights


def run_simulation(config):
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    device      = get_device()
    results_dir = config["logging"].get("results_dir", "results")
    make_dirs(results_dir)

    dataset_name = config["dataset"]["name"]
    train_dataset, test_dataset = load_dataset(dataset_name)

    num_clients = config["federation"]["num_clients"]
    partition   = config["dataset"]["partition"]
    alpha       = config["dataset"].get("dirichlet_alpha", 0.5) or 0.5
    client_indices = partition_data(train_dataset, num_clients, partition, alpha, seed)

    exp_name          = config["experiment"]["name"]
    algorithm         = config["experiment"]["algorithm"].lower()
    _tmp_model        = get_model(config["model"]["architecture"], config["dataset"]["num_classes"])
    model_size_mb     = sum(p.numel() for p in _tmp_model.parameters()) * 4 / (1024 ** 2)
    clients_per_round = max(1, int(config["federation"]["fraction_fit"] * num_clients))

    logger = MetricsLogger(
        experiment_name=exp_name, results_dir=results_dir,
        convergence_threshold=80.0, model_size_mb=model_size_mb,
        num_clients_per_round=clients_per_round,
    )

    global_weights = get_parameters(
        get_model(config["model"]["architecture"], config["dataset"]["num_classes"]).to(device)
    )

    # Instantiate clients based on algorithm
    if algorithm == "fedasync":
        clients = [FedAsyncClient(cid, train_dataset, test_dataset,
                                  client_indices[cid], config, device)
                   for cid in range(num_clients)]
    elif algorithm == "fedcs":
        clients = [FedCSClient(cid, train_dataset, test_dataset,
                               client_indices[cid], config, device)
                   for cid in range(num_clients)]
    else:
        clients = [FedAvgClient(cid, train_dataset, test_dataset,
                                client_indices[cid], config, device)
                   for cid in range(num_clients)]

    num_rounds = config["federation"]["num_rounds"]

    print(f"\n{'='*65}")
    print(f"  Experiment  : {exp_name}")
    print(f"  Algorithm   : {algorithm.upper()}")
    print(f"  Dataset     : {dataset_name.upper()}")
    print(f"  Partition   : {partition}" + (f"  (α={alpha})" if partition == "noniid_dirichlet" else ""))
    print(f"  Clients     : {num_clients}  |  Sampled/round: {clients_per_round}")
    print(f"  Rounds      : {num_rounds}  |  Local Epochs: {config['federation']['local_epochs']}")
    print(f"  Batch Size  : {config['federation']['batch_size']}  |  LR: {config['optimizer']['lr']}")
    print(f"  Model size  : {model_size_mb:.3f} MB")
    print(f"{'='*65}\n")

    if algorithm == "fedasync":
        run_fedasync(config, clients, global_weights, logger, num_rounds, device, seed)
    elif algorithm == "fedcs":
        run_fedcs(config, clients, global_weights, logger, num_rounds, model_size_mb, device, seed)
    else:
        run_fedavg(config, clients, global_weights, logger, num_rounds, seed)

    logger.save()
    if config["logging"].get("save_results", True) and len(logger.records) > 0:
        plot_metrics(exp_name, results_dir)
        print(f"[Done] Plots saved to {results_dir}/plots/")
    print(f"\n[Done] '{exp_name}' -> results saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Simulation — FedAvg / FedAsync / FedCS")
    parser.add_argument("--config", type=str, required=True)
    args   = parser.parse_args()
    run_simulation(load_config(args.config))
