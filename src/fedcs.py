"""
fedcs.py — Client Selection for Federated Learning (FedCS)

Paper: "Client Selection for Federated Learning with Heterogeneous
        Resources in Mobile Edge"
       Nishio & Yonetani. IEEE ICC 2019. arXiv:1804.08333.

Core idea:
  Instead of random client selection (FedAvg), FedCS selects clients
  based on their RESOURCE CAPABILITY subject to a round DEADLINE T_round.
  It greedily maximises the number of clients that can finish within T_round.

FedCS Protocol (Algorithm 2 in paper):
  Step 1 — Resource Request:
    Server asks a random subset of K' clients for their resource info:
    - compute capability θ_k  (samples/second)
    - data size d_k           (number of local samples)
    - upload speed u_k        (MB/second)

  Step 2 — Client Selection:
    Estimate completion time for each candidate:
      t_k = (d_k / θ_k) * local_epochs + model_size_mb / u_k
    Greedily select max clients where t_k <= T_round
    (sorted by t_k ascending — fastest first)

  Step 3 — Standard FedAvg on selected clients

Simulation of heterogeneous resources:
  Since we run on one machine, each client gets simulated:
  - compute_speed θ_k  : samples/second (drawn from distribution)
  - upload_speed u_k   : MB/second (drawn from distribution)
  These determine estimated_time_k and straggler classification.
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict
import torch

from src.client import FedAvgClient


# ──────────────────────────────────────────────
#  FedCS Client — adds resource profile
# ──────────────────────────────────────────────

class FedCSClient(FedAvgClient):
    """
    FedCS client with simulated resource profile.
    Each client has:
      - compute_speed: samples/second (higher = faster)
      - upload_speed : MB/second (higher = faster)
    These are used by the server to estimate completion time.
    """

    def __init__(self, client_id, train_dataset, val_dataset,
                 client_indices, config, device):
        super().__init__(client_id, train_dataset, val_dataset,
                         client_indices, config, device)

        fedcs_cfg = config.get("fedcs", {})
        rng       = random.Random(config["experiment"]["seed"] + client_id + 1000)

        speed_dist = fedcs_cfg.get("speed_distribution", "uniform")

        if speed_dist == "uniform":
            self.compute_speed = rng.uniform(0.5, 2.0)    # samples/sec (relative)
            self.upload_speed  = rng.uniform(1.0, 5.0)    # MB/sec (relative)

        elif speed_dist == "heterogeneous":
            # Mix of fast and slow clients — realistic MEC scenario
            device_type = rng.choices(
                ["fast", "medium", "slow"],
                weights=[0.3, 0.4, 0.3]
            )[0]
            if device_type == "fast":
                self.compute_speed = rng.uniform(1.5, 3.0)
                self.upload_speed  = rng.uniform(3.0, 6.0)
            elif device_type == "medium":
                self.compute_speed = rng.uniform(0.8, 1.5)
                self.upload_speed  = rng.uniform(1.5, 3.0)
            else:  # slow
                self.compute_speed = rng.uniform(0.1, 0.5)
                self.upload_speed  = rng.uniform(0.3, 1.0)

        else:
            self.compute_speed = 1.0
            self.upload_speed  = 2.0

    def estimated_completion_time(
        self,
        local_epochs: int,
        model_size_mb: float,
    ) -> float:
        """
        Estimate time for this client to complete one round.
        t_k = (num_samples / compute_speed) * local_epochs + model_size_mb / upload_speed

        This is the core of FedCS client selection.
        """
        compute_time = (len(self.trainloader.dataset) / self.compute_speed) * local_epochs
        upload_time  = model_size_mb / self.upload_speed
        return compute_time + upload_time

    def get_resource_info(self, local_epochs: int, model_size_mb: float) -> dict:
        """Return resource profile — sent to server in Resource Request step."""
        return {
            "client_id":       self.client_id,
            "compute_speed":   self.compute_speed,
            "upload_speed":    self.upload_speed,
            "num_samples":     len(self.trainloader.dataset),
            "estimated_time":  self.estimated_completion_time(local_epochs, model_size_mb),
        }


# ──────────────────────────────────────────────
#  FedCS Selection Algorithm
# ──────────────────────────────────────────────

def fedcs_select_clients(
    candidate_resources: List[dict],
    t_round: float,
    min_clients: int = 1,
) -> List[int]:
    """
    FedCS greedy client selection (Algorithm 2, Step 2).

    Sort candidates by estimated_time ascending.
    Select all clients whose estimated_time <= T_round.
    If none qualify, fall back to min_clients fastest.

    Args:
        candidate_resources : list of resource dicts from get_resource_info()
        t_round             : round deadline in seconds (simulated)
        min_clients         : minimum clients to select even if over deadline

    Returns:
        List of selected client_ids
    """
    # Sort by estimated completion time (fastest first)
    sorted_candidates = sorted(
        candidate_resources, key=lambda x: x["estimated_time"]
    )

    selected = [
        r["client_id"] for r in sorted_candidates
        if r["estimated_time"] <= t_round
    ]

    # Fallback: if no client fits deadline, take the min_clients fastest
    if len(selected) < min_clients:
        selected = [r["client_id"] for r in sorted_candidates[:min_clients]]

    return selected


# ──────────────────────────────────────────────
#  FedCS Aggregation (standard FedAvg)
# ──────────────────────────────────────────────

def fedcs_aggregate(results: List[Tuple[int, List[np.ndarray]]]) -> List[np.ndarray]:
    """Standard FedAvg weighted aggregation — same as baseline."""
    total_samples = sum(n for n, _ in results)
    aggregated    = [np.zeros_like(w) for w in results[0][1]]
    for num_samples, weights in results:
        for i, w in enumerate(weights):
            aggregated[i] += w * (num_samples / total_samples)
    return aggregated


# ──────────────────────────────────────────────
#  FedCS Simulation Runner
# ──────────────────────────────────────────────

def run_fedcs(
    config: dict,
    clients: List[FedCSClient],
    global_weights: List[np.ndarray],
    logger,
    num_rounds: int,
    model_size_mb: float,
    device: torch.device,
    seed: int = 42,
):
    """
    Run FedCS simulation.

    FedCS Protocol each round:
      1. Resource Request: ask K' random clients for resource info
      2. Client Selection: greedily select by deadline T_round
      3. Train selected clients (FedAvg local training)
      4. Aggregate (FedAvg weighted average)
      5. Evaluate + log
    """
    fedcs_cfg        = config.get("fedcs", {})
    t_round          = fedcs_cfg.get("t_round", 500.0)        # simulated deadline
    request_fraction = fedcs_cfg.get("request_fraction", 0.5) # K'/K
    local_epochs     = config["federation"]["local_epochs"]
    num_clients      = len(clients)
    rng              = random.Random(seed)
    straggler_thr    = fedcs_cfg.get("straggler_threshold", 200.0)  # time threshold

    for rnd in range(1, num_rounds + 1):
        round_start = time.time()

        # ── Step 1: Resource Request ──
        # Ask random subset of clients for their resource info
        num_request    = max(1, int(request_fraction * num_clients))
        request_cids   = rng.sample(range(num_clients), num_request)
        resource_infos = [
            clients[cid].get_resource_info(local_epochs, model_size_mb)
            for cid in request_cids
        ]

        # ── Step 2: Client Selection ──
        # Greedily select clients that can finish within T_round
        min_clients   = max(1, int(config["federation"]["fraction_fit"] * num_clients))
        selected_ids  = fedcs_select_clients(resource_infos, t_round, min_clients)

        # ── Step 3: Train selected clients ──
        fit_results = []
        for cid in selected_ids:
            weights, n_samples, _ = clients[cid].fit(
                parameters=global_weights, config={}
            )
            fit_results.append((n_samples, weights))

        # ── Step 4: Aggregate ──
        if fit_results:
            global_weights = fedcs_aggregate(fit_results)

        # ── Step 5: Evaluate ──
        eval_losses, eval_accs, eval_counts = [], [], []
        for cid in selected_ids:
            loss, n, metrics = clients[cid].evaluate(
                parameters=global_weights, config={}
            )
            eval_losses.append(loss * n)
            eval_accs.append(metrics["val_accuracy"] * n)
            eval_counts.append(n)

        total       = sum(eval_counts)
        global_loss = sum(eval_losses) / total if total > 0 else 0.0
        global_acc  = sum(eval_accs)   / total if total > 0 else 0.0
        round_time  = round(time.time() - round_start, 3)

        # Straggler ratio: fraction of REQUESTED clients that were NOT selected
        # (they were too slow to meet the deadline)
        not_selected  = set(request_cids) - set(selected_ids)
        strag_ratio   = len(not_selected) / len(request_cids) if request_cids else 0.0

        print(f"[FedCS Round {rnd:>3}/{num_rounds}]  "
              f"Loss: {global_loss:.4f}  |  "
              f"Acc: {global_acc:.2f}%  |  "
              f"Time: {round_time:.1f}s  |  "
              f"Selected: {len(selected_ids)}/{num_request} requested  |  "
              f"Straggler ratio: {strag_ratio:.2f}")

        logger.log(
            round_num=rnd,
            global_test_accuracy=global_acc,
            global_test_loss=global_loss,
            round_completion_time=round_time,
            straggler_ratio=strag_ratio,
            participated_clients=len(selected_ids),
        )
        logger.save()

    return global_weights
