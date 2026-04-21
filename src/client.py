"""
client.py — Flower client implementing local training for FedAvg.

FedAvg Spec (McMahan et al., 2017):
  - Local Epochs     : 5
  - Batch Size       : 32
  - Optimizer        : SGD (momentum=0.9)
  - Learning Rate    : 0.01
  - Loss             : Cross-Entropy
  - Weight Init      : PyTorch default
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.model import get_model
from src.data import get_client_dataloader


# ──────────────────────────────────────────────
#  Helper: model ↔ numpy parameter conversion
# ──────────────────────────────────────────────

def get_parameters(model: nn.Module) -> NDArrays:
    """Extract model weights as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    """Load a list of numpy arrays into the model's state dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# ──────────────────────────────────────────────
#  FedAvg Flower Client
# ──────────────────────────────────────────────

class FedAvgClient(fl.client.NumPyClient):
    """
    Flower client for FedAvg baseline.

    Each client:
      1. Receives global model weights from the server.
      2. Trains locally for E epochs on its own data.
      3. Returns updated weights + metrics to the server.
    """

    def __init__(
        self,
        client_id: int,
        train_dataset,
        val_dataset,
        client_indices: List[int],
        config: dict,
        device: torch.device,
    ):
        self.client_id = client_id
        self.config = config
        self.device = device

        # ── Model ──
        self.model = get_model(
            architecture=config["model"]["architecture"],
            num_classes=config["dataset"]["num_classes"],
        ).to(device)

        # ── DataLoaders ──
        fed_cfg = config["federation"]
        self.trainloader = get_client_dataloader(
            train_dataset,
            client_indices,
            batch_size=fed_cfg["batch_size"],
            shuffle=True,
        )
        self.valloader = get_client_dataloader(
            val_dataset,
            list(range(len(val_dataset))),   # full test set for eval
            batch_size=64,
            shuffle=False,
        )

        # ── Loss ──
        self.criterion = nn.CrossEntropyLoss()

        # ── Optimizer (SGD, momentum=0.9, lr=0.01 per FedAvg spec) ──
        opt_cfg = config["optimizer"]
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg["momentum"],
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )

    # ── Flower API ──────────────────────────────

    def get_parameters(self, config: Dict) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global weights → local train → return updated weights.
        """
        # 1. Load global model weights
        set_parameters(self.model, parameters)

        local_epochs = self.config["federation"]["local_epochs"]

        # 2. Local training
        train_loss, train_acc = self._train(local_epochs)

        # 3. Return updated weights + metrics
        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "client_id": self.client_id,
            },
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Receive global weights → evaluate on local val set → return metrics.
        """
        set_parameters(self.model, parameters)
        loss, accuracy = self._evaluate()
        return (
            float(loss),
            len(self.valloader.dataset),
            {"val_accuracy": float(accuracy), "client_id": self.client_id},
        )

    # ── Internal training / evaluation ──────────

    def _train(self, epochs: int) -> Tuple[float, float]:
        """Run local SGD for `epochs` epochs."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate on the validation/test set."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy