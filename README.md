# Federated Learning Experiments

This repository contains a federated learning project for MNIST and CIFAR-10 using Flower and PyTorch.

## Project Overview

- **Category:** Federated Learning / Distributed Machine Learning
- **Group Info:**
  - Group Name: _[Add group name here]_
  - Members:
    - _[Member 1 Name]_
    - _[Member 2 Name]_
    - _[Member 3 Name]_
    - _[Member 4 Name]_
    - _[Member 5 Name]_
    - _[Member 6 Name]_
- **Paper Titles:**
  - _[Add primary paper title here]_
  - _[Add additional paper titles here]_
  - _[Add additional paper titles here]_
  
- **YouTube Video:** _[Add YouTube presentation URL here]_

## Setup Instructions

### Linux / macOS

1. Open a terminal and change to the repository root:
   ```bash
   cd "/home/manish-singh/Manish Singh/Mtech Coursework/Wireless"
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Windows (PowerShell)

1. Open PowerShell and change to the repository root:
   ```powershell
   cd "C:\Users\<your-user>\path\to\Wireless"
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Running Experiments

### Run a single config

```bash
python main.py --config configs/mnist_fedavg_10clients_iid.yaml
```

### Run all experiments and generate final results

First make sure `run_all_experiments.sh` is executable:

```bash
chmod +x run_all_experiments.sh
```

Then run:

```bash
bash run_all_experiments.sh
```

This script will:
- execute each config file in `configs/`
- save results to `results/metrics/`
- generate final plots and summary tables in `results/plots/`

### Generate plots and summary table manually

```bash
python generate_figures.py --results-dir results --proposed fedasync
```

## Results

- Metrics are saved to `results/metrics/`
- Plots are saved to `results/plots/`
- Summary table is saved to `results/metrics/results_table.csv`

## Notes

- If you are running on Windows and PowerShell blocks script execution, you may need to allow script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- Update the placeholders above with your group details, category, papers, and video link.
