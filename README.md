pip install -r requirements.txt

python main.py --config configs/mnist_fedavg_10clients_iid.yaml

# after running all the experiments type these commands
# from src.utils import generate_all_figures
# generate_all_figures(results_dir="results")