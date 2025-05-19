import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.config import prob_matrix, alphas, kernel_function, kernel_params, final_time
from src.synthetic_data import generate_sbm_adjacency_matrix, expand_adjacency_matrix
from src.simulation import simulate_comb_sbm, simulate_mf_sbm
from src.ploting import plot_N

two_norms = []
infinity_norms = []
max_iter = 5000
min_iter = 100
iter_by= 5

range_values = np.arange(min_iter, max_iter, iter_by)

num_nodes = min_iter
comm_size_prev = (num_nodes * alphas).astype(int)
num_nodes = np.sum(comm_size_prev)
labs, G = generate_sbm_adjacency_matrix(comm_size_prev, prob_matrix)


for value in tqdm(range_values):
    num_nodes = value
    comm_size_new = (num_nodes * alphas).astype(int)
    num_nodes = np.sum(comm_size_new)

    E = np.random.exponential(scale = 1.0, size = (final_time, num_nodes, 15))
    labs, G = expand_adjacency_matrix(labs, G, comm_size_prev, comm_size_new, prob_matrix)

    # Simulate the combined process and mean-field approximation
    N_comb, lmbd_comb ,X_comb = simulate_comb_sbm(E, final_time,num_nodes,comm_size_new, kernel_function, kernel_params, G)
    N_mf, lmbd_mf, X_mf = simulate_mf_sbm(E, final_time, comm_size_new, prob_matrix, kernel_function, kernel_params)

    start_idx = 0
    lmbd_mf_expanded = np.zeros_like(lmbd_comb)
    for i, size in enumerate(comm_size_new):
            lmbd_mf_expanded[:, start_idx:start_idx + size] = np.tile(lmbd_mf[:, i].reshape(-1, 1), size)
            start_idx += size

    comm_size_prev = comm_size_new
    two_norm = np.linalg.norm(np.sum(np.abs(lmbd_comb - lmbd_mf_expanded), axis = 0), ord=2)/final_time
    infinity_norm = np.linalg.norm(np.abs(lmbd_comb - lmbd_mf_expanded), ord=np.inf) 

    two_norms.append(two_norm)
    infinity_norms.append(infinity_norm)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

results_dir = f"results/N_dep_{kernel_function}_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

np.save(f"{results_dir}/two_norms.npy", two_norms)
np.save(f"{results_dir}/infty_norms.npy", infinity_norms)

plot_N(range_values, two_norms, results_dir)


