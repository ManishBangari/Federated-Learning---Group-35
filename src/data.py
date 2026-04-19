"""
data.py — Dataset loading and client-side data partitioning.

Supports:
  - MNIST
  - CIFAR-10

Partitioning strategies:
  - IID     : shuffle and split equally across clients
  - Non-IID : Dirichlet distribution (alpha controls heterogeneity)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


# ──────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────

TRANSFORMS = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "fmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ]),
}


# ──────────────────────────────────────────────
#  Dataset Loader
# ──────────────────────────────────────────────

def load_dataset(dataset_name: str, data_dir: str = "./data"):
    """
    Download and return (train_dataset, test_dataset).
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        train = datasets.MNIST(
            root=data_dir, train=True, download=True,
            transform=TRANSFORMS["mnist"]
        )
        test = datasets.MNIST(
            root=data_dir, train=False, download=True,
            transform=TRANSFORMS["mnist"]
        )

    elif dataset_name == "fmnist":
        train = datasets.FashionMNIST(
            root=data_dir, train=True, download=True,
            transform=TRANSFORMS["fmnist"]
        )
        test = datasets.FashionMNIST(
            root=data_dir, train=False, download=True,
            transform=TRANSFORMS["fmnist"]
        )

    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(
            root=data_dir, train=True, download=True,
            transform=TRANSFORMS["cifar10"]
        )
        test = datasets.CIFAR10(
            root=data_dir, train=False, download=True,
            transform=TRANSFORMS["cifar10"]
        )

    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. "
                         f"Choose from: mnist, fmnist, cifar10")

    print(f"[Data] Loaded {dataset_name.upper()} — "
          f"Train: {len(train)} | Test: {len(test)}")
    return train, test


# ──────────────────────────────────────────────
#  IID Partitioning
# ──────────────────────────────────────────────

def iid_partition(
    dataset,
    num_clients: int,
    seed: int = 42
) -> List[List[int]]:
    """
    Split dataset indices equally and randomly across clients (IID).

    Returns:
        List of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    client_indices = [split.tolist() for split in splits]

    print(f"[Partition] IID — {num_clients} clients, "
          f"~{len(client_indices[0])} samples each")
    return client_indices


# ──────────────────────────────────────────────
#  Non-IID Partitioning (Dirichlet)
# ──────────────────────────────────────────────

def noniid_dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Partition dataset using Dirichlet distribution.

    Lower alpha → more heterogeneous (extreme non-IID).
    Higher alpha → more uniform (approaches IID).

    Returns:
        List of index lists, one per client.
    """
    rng = np.random.default_rng(seed)

    # Extract labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Assign samples to clients proportionally
        proportions = (proportions * len(cls_idx)).astype(int)

        # Fix rounding to ensure all samples are assigned
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()

        start = 0
        for client_id, count in enumerate(proportions):
            end = start + count
            client_indices[client_id].extend(cls_idx[start:end].tolist())
            start = end

    # Shuffle each client's data
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    sizes = [len(c) for c in client_indices]
    print(f"[Partition] Non-IID Dirichlet (α={alpha}) — "
          f"{num_clients} clients | "
          f"min={min(sizes)}, max={max(sizes)}, mean={int(np.mean(sizes))}")
    return client_indices


# ──────────────────────────────────────────────
#  DataLoader Builder
# ──────────────────────────────────────────────

def get_client_dataloader(
    dataset,
    indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Return a DataLoader for a specific client's data subset."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_test_dataloader(
    dataset,
    batch_size: int = 64
) -> DataLoader:
    """Return a DataLoader for the global test set."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ──────────────────────────────────────────────
#  Unified Partition Entry Point
# ──────────────────────────────────────────────

def partition_data(
    dataset,
    num_clients: int,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Unified partitioning interface.

    Args:
        dataset       : torchvision dataset
        num_clients   : total number of FL clients
        partition     : 'iid' or 'noniid_dirichlet'
        dirichlet_alpha: alpha for Dirichlet (ignored if IID)
        seed          : random seed

    Returns:
        List of index lists, one per client
    """
    if partition == "iid":
        return iid_partition(dataset, num_clients, seed)
    elif partition == "noniid_dirichlet":
        return noniid_dirichlet_partition(
            dataset, num_clients, dirichlet_alpha, seed
        )
    else:
        raise ValueError(f"Unknown partition: '{partition}'. "
                         f"Use 'iid' or 'noniid_dirichlet'.")