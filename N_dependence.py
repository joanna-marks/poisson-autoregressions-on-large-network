import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.config import prob_matrix, alphas, final_time, kernel_function_matrix, kernel_params_matrix
from src.synthetic_data import generate_sbm_adjacency_matrix, expand_adjacency_matrix
from src.simulation import simulate_comb_sbm, simulate_mf_sbm, simulate_aux_sbm
from src.ploting import plot_N
from src.helper_functions import exponential_kernel

kernel_function = exponential_kernel


two_norms_comb = []
infinity_norms_comb = []
two_norms_aux = []
infinity_norms_aux = []
mean_distances_lmbd = []
mean_distances_X = []
max_iter = 1000
min_iter = 100
iter_by= 50

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
    N_comb, lmbd_comb ,X_comb = simulate_comb_sbm(E, final_time, num_nodes, comm_size_new, kernel_function_matrix, kernel_params_matrix, G, labs)
    N_aux, lmbd_aux ,X_aux = simulate_aux_sbm(E, final_time, num_nodes, comm_size_new, kernel_function_matrix, kernel_params_matrix, G, labs)
    N_mf, lmbd_mf, X_mf = simulate_mf_sbm(E, final_time, comm_size_new, prob_matrix, kernel_function_matrix, kernel_params_matrix)

    start_idx = 0
    lmbd_mf_expanded = np.zeros_like(lmbd_comb)
    for i, size in enumerate(comm_size_new):
            lmbd_mf_expanded[:, start_idx:start_idx + size] = np.tile(lmbd_mf[:, i].reshape(-1, 1), size)
            start_idx += size

    two_norm_comb = np.linalg.norm(np.abs(lmbd_comb[-1,:] - lmbd_aux[-1,:]), ord=2)/final_time
    infinity_norm_comb = np.linalg.norm(np.abs(lmbd_comb[-1,:] - lmbd_aux[-1,:]), ord=np.inf) 
    mean_dist_lmbd = np.mean(np.abs(lmbd_comb[-1,:] - lmbd_mf_expanded[-1,:]))
    mean_dist_X = np.mean(np.abs(X_comb[-1,:] - X_mf[-1,:]))

    

    for i in range(20):
        E = np.random.exponential(scale = 1.0, size = (final_time, num_nodes, 15))
        # Simulate the combined process and mean-field approximation
        N_comb_new, lmbd_comb_new ,X_comb_new = simulate_comb_sbm(E, final_time, num_nodes, comm_size_new, kernel_function_matrix, kernel_params_matrix, G, labs)
        two_norm_comb_new = np.linalg.norm(np.abs(lmbd_comb_new[-1,:] - lmbd_aux[-1,:]), ord=2)/final_time
        infinity_norm_comb_new = np.linalg.norm(np.abs(lmbd_comb_new[-1,:] - lmbd_aux[-1,:]), ord=np.inf) 

        start_idx = 0
        lmbd_mf_expanded = np.zeros_like(lmbd_comb)
        for i, size in enumerate(comm_size_new):
            lmbd_mf_expanded[:, start_idx:start_idx + size] = np.tile(lmbd_mf[:, i].reshape(-1, 1), size)
            start_idx += size

        two_norm_comb = (two_norm_comb + two_norm_comb_new)/2
        infinity_norm_comb = (infinity_norm_comb + infinity_norm_comb_new)/2
        mean_dist_lmbd = (mean_dist_lmbd + np.mean(np.abs(lmbd_comb_new[-1,:] - lmbd_mf_expanded[-1,:])))/2
        mean_dist_X = (mean_dist_X + np.mean(np.abs(X_comb_new[-1,:] - X_mf[-1,:])))/2


   

    comm_size_prev = comm_size_new
    # two_norm_comb = np.linalg.norm(np.abs(lmbd_comb[-1,:] - lmbd_aux[-1,:]), ord=2)/final_time
    # infinity_norm_comb = np.linalg.norm(np.abs(lmbd_comb[-1,:] - lmbd_aux[-1,:]), ord=np.inf) 

    two_norm_aux = np.linalg.norm(lmbd_aux[-1,:] - lmbd_mf_expanded[-1,:], ord=2)/final_time
    infinity_norm_aux= np.linalg.norm(np.abs(lmbd_aux[-1,:] - lmbd_mf_expanded[-1,:]), ord=np.inf) 

    two_norms_comb.append(two_norm_comb)
    infinity_norms_comb.append(infinity_norm_comb)

    two_norms_aux.append(two_norm_aux)
    infinity_norms_aux.append(infinity_norm_aux)
    mean_distances_lmbd.append(mean_dist_lmbd)
    mean_distances_X.append(mean_dist_X)
    print(f"Nodes: {num_nodes}, Two-norm (combined): {two_norm_comb}, Infinity-norm (combined): {infinity_norm_comb}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

results_dir = f"results/N_dep_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

np.save(f"{results_dir}/two_norms_comb.npy", two_norms_comb)
np.save(f"{results_dir}/infty_norms_comb.npy", infinity_norms_comb)
np.save(f"{results_dir}/two_norms_aux.npy", two_norms_aux)
np.save(f"{results_dir}/infty_norms_aux.npy", infinity_norms_aux)
np.save(f"{results_dir}/mean_distances.npy", mean_distances_lmbd)
np.save(f"{results_dir}/mean_distances_X.npy", mean_distances_X)


print("two_norms_comb shape:", np.shape(two_norms_comb))
print("infinity_norms_comb shape:", np.shape(infinity_norms_comb))
print("two_norms_aux shape:", np.shape(two_norms_aux))
print("infinity_norms_aux shape:", np.shape(infinity_norms_aux))
print("mean_distances shape:", np.shape(mean_distances_lmbd))
print("mean_distances_X shape:", np.shape(mean_distances_X))

#plot_N(range_values, two_norms_comb, infinity_norms_comb, results_dir)
#plot_N(range_values, two_norms_aux, infinity_norms_aux, results_dir)


