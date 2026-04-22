"""
fedasync.py — Asynchronous Federated Optimization (FedAsync)

Paper: "Asynchronous Federated Optimization"
       Xie, Koyejo, Gupta. arXiv:1903.03934, 2019.

Core idea:
  Server updates global model IMMEDIATELY when any client returns,
  without waiting for all clients. Stale updates are downweighted
  using a staleness function.

Update rule (Eq. 2 in paper):
  w_new = (1 - alpha_t) * w_global + alpha_t * w_client
  alpha_t = mixing_rate * staleness_fn(tau)

  where tau = staleness = how many server updates happened since
  this client last pulled the global model.

Staleness functions (Section 5.2):
  - constant   : s(tau) = 1
  - polynomial : s(tau) = 1 / (1 + tau)^a       (a=0.5 default)
  - hinge      : s(tau) = 1 if tau<=b else 1/(a*(tau-b)+1)
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Optional
import torch

from src.model import get_model, get_parameters
from src.client import FedAvgClient


# ──────────────────────────────────────────────
#  Staleness Functions
# ──────────────────────────────────────────────

def staleness_constant(tau: int, a: float = 0.5, b: int = 4) -> float:
    """s(tau) = 1  — no penalty for staleness."""
    return 1.0


def staleness_polynomial(tau: int, a: float = 0.5, b: int = 4) -> float:
    """s(tau) = 1 / (1 + tau)^a  — polynomial decay."""
    return 1.0 / ((1 + tau) ** a)


def staleness_hinge(tau: int, a: float = 0.5, b: int = 4) -> float:
    """
    s(tau) = 1           if tau <= b
           = 1/(a*(tau-b)+1)  if tau > b
    """
    if tau <= b:
        return 1.0
    return 1.0 / (a * (tau - b) + 1.0)


STALENESS_FNS = {
    "constant":   staleness_constant,
    "polynomial": staleness_polynomial,
    "hinge":      staleness_hinge,
}


# ──────────────────────────────────────────────
#  FedAsync Client (extends FedAvgClient)
# ──────────────────────────────────────────────

class FedAsyncClient(FedAvgClient):
    """
    FedAsync client. Identical to FedAvg client for local training —
    the async logic is entirely on the server side.
    Additionally tracks which global model version it trained on.
    """

    def __init__(self, client_id, train_dataset, val_dataset,
                 client_indices, config, device):
        super().__init__(client_id, train_dataset, val_dataset,
                         client_indices, config, device)
        # Version of global model this client last trained on
        self.last_global_version = 0

        # Simulated per-client compute speed (for straggler simulation)
        # Drawn once at init — slower clients take longer to "respond"
        rng = random.Random(config["experiment"]["seed"] + client_id)
        speed_dist = config.get("fedasync", {}).get("speed_distribution", "uniform")
        if speed_dist == "uniform":
            self.compute_speed = rng.uniform(0.5, 1.5)   # relative speed multiplier
        elif speed_dist == "heterogeneous":
            # Some clients are much slower — simulates real-world heterogeneity
            self.compute_speed = rng.choice([0.2, 0.5, 1.0, 1.0, 2.0])
        else:
            self.compute_speed = 1.0

    def fit_with_version(self, parameters, config, global_version: int):
        """
        Train locally and return results along with the version
        of the global model that was used for training.
        """
        self.last_global_version = global_version
        weights, num_samples, metrics = self.fit(parameters, config)
        metrics["global_version_used"] = global_version
        return weights, num_samples, metrics


# ──────────────────────────────────────────────
#  FedAsync Server Logic
# ──────────────────────────────────────────────

class FedAsyncServer:
    """
    Asynchronous FL server (Xie et al., 2019).

    In each "async step":
      - One client returns its update (simulated by random ordering)
      - Server immediately updates global model using staleness-weighted mixing
      - global_version increments by 1

    Staleness τ = global_version_now - global_version_when_client_trained
    """

    def __init__(self, config: dict, global_weights: List[np.ndarray]):
        async_cfg           = config.get("fedasync", {})
        self.mixing_rate    = async_cfg.get("mixing_rate", 0.1)
        self.staleness_fn   = STALENESS_FNS[
            async_cfg.get("staleness_fn", "polynomial")
        ]
        self.staleness_a    = async_cfg.get("staleness_a", 0.5)
        self.staleness_b    = async_cfg.get("staleness_b", 4)
        self.global_weights = [w.copy() for w in global_weights]
        self.global_version = 0

    def async_update(
        self,
        client_weights: List[np.ndarray],
        client_version: int,
        num_samples: int,
    ) -> Tuple[List[np.ndarray], float, int]:
        """
        Apply one asynchronous update from a single client.

        Returns: (new_global_weights, alpha_used, staleness)
        """
        tau      = self.global_version - client_version
        s_tau    = self.staleness_fn(tau, self.staleness_a, self.staleness_b)
        alpha_t  = self.mixing_rate * s_tau

        # w_new = (1 - alpha_t) * w_global + alpha_t * w_client
        new_weights = [
            (1 - alpha_t) * gw + alpha_t * cw
            for gw, cw in zip(self.global_weights, client_weights)
        ]
        self.global_weights = new_weights
        self.global_version += 1

        return new_weights, alpha_t, tau


# ──────────────────────────────────────────────
#  FedAsync Simulation Runner
# ──────────────────────────────────────────────

def run_fedasync(
    config: dict,
    clients: List[FedAsyncClient],
    global_weights: List[np.ndarray],
    logger,
    num_rounds: int,
    device: torch.device,
    seed: int = 42,
):
    """
    Run FedAsync simulation.

    Simulates async by:
      1. Each "round" = N async steps where N = num_clients
      2. In each step, one randomly chosen client trains on the
         CURRENT global model and returns immediately
      3. Server updates using staleness-weighted mixing
      4. After N steps, evaluate global model = one "epoch" for logging

    Straggler simulation:
      - Each client has a compute_speed attribute
      - Clients are ordered by simulated arrival time = base_time / speed
      - Stragglers (speed < threshold) are counted for straggler_ratio
    """
    server        = FedAsyncServer(config, global_weights)
    num_clients   = len(clients)
    rng           = random.Random(seed)
    async_cfg     = config.get("fedasync", {})
    straggler_thr = async_cfg.get("straggler_threshold", 0.5)
    steps_per_rnd = async_cfg.get("steps_per_round", num_clients)

    for rnd in range(1, num_rounds + 1):
        round_start   = time.time()
        stragglers    = 0
        participated  = 0

        # Simulate async arrivals: shuffle clients by arrival time
        arrival_order = sorted(
            range(num_clients),
            key=lambda cid: rng.random() / clients[cid].compute_speed
        )

        for step_idx in range(min(steps_per_rnd, num_clients)):
            cid            = arrival_order[step_idx]
            client_version = server.global_version

            # Client trains on current global model
            c_weights, n_samples, _ = clients[cid].fit_with_version(
                parameters=server.global_weights,
                config={},
                global_version=client_version,
            )

            # Async server update
            server.async_update(c_weights, client_version, n_samples)

            # Track stragglers (slow clients)
            if clients[cid].compute_speed < straggler_thr:
                stragglers += 1
            participated += 1

        # ── Evaluate global model on sampled clients ──
        eval_cids = rng.sample(range(num_clients),
                               min(steps_per_rnd, num_clients))
        losses, accs, counts = [], [], []
        for cid in eval_cids:
            loss, n, metrics = clients[cid].evaluate(
                parameters=server.global_weights, config={}
            )
            losses.append(loss * n)
            accs.append(metrics["val_accuracy"] * n)
            counts.append(n)

        total       = sum(counts)
        global_loss = sum(losses) / total
        global_acc  = sum(accs)   / total
        round_time  = round(time.time() - round_start, 3)
        strag_ratio = stragglers / participated if participated > 0 else 0.0

        print(f"[FedAsync Round {rnd:>3}/{num_rounds}]  "
              f"Loss: {global_loss:.4f}  |  "
              f"Acc: {global_acc:.2f}%  |  "
              f"Time: {round_time:.1f}s  |  "
              f"Straggler ratio: {strag_ratio:.2f}  |  "
              f"GlobalVer: {server.global_version}")

        logger.log(
            round_num=rnd,
            global_test_accuracy=global_acc,
            global_test_loss=global_loss,
            round_completion_time=round_time,
            straggler_ratio=strag_ratio,
            participated_clients=participated,
        )
        logger.save()

    return server.global_weights
