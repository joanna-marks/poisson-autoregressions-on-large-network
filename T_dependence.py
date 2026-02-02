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
from src.helper_functions import exponential_kernel

kernel_function = exponential_kernel

num_nodes =50
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


num_graphs = 5
# Lists to hold one list per graph
two_norms_comb_all = [[] for _ in range(num_graphs)]
infinity_norms_comb_all = [[] for _ in range(num_graphs)]
two_norms_aux_all = [[] for _ in range(num_graphs)]
infinity_norms_aux_all = [[] for _ in range(num_graphs)]
mean_distances_lmbd_all = [[] for _ in range(num_graphs)]
mean_distances_X_all = [[] for _ in range(num_graphs)]
max_distances_X_all =  [[] for _ in range(num_graphs)]
max_distances_lmbd_all =  [[] for _ in range(num_graphs)]

for graph_idx in range(num_graphs):
    np.random.seed(graph_idx * 1000)
    print(f"\nGraph Realization {graph_idx + 1}/{num_graphs}")
    final_time = min_iter
    comm_size = (num_nodes * alphas).astype(int)
    labs, G = generate_sbm_adjacency_matrix(comm_size, prob_matrix)

    for value in tqdm(range_values, desc=f"Graph {graph_idx+1} progress"):
        final_time = value

        E = np.random.exponential(scale=1.0, size=(final_time, num_nodes, 50))

        # Base simulations
        N_aux, lmbd_aux, X_aux = simulate_aux_sbm(E, final_time, num_nodes, comm_size, kernel_function_matrix, kernel_params_matrix, G, labs)
        N_mf, lmbd_mf, X_mf = simulate_mf_sbm(E, final_time, comm_size, prob_matrix, kernel_function_matrix, kernel_params_matrix)

        # E
        lmbd_mf_expanded = np.zeros((final_time, num_nodes))
        start_idx = 0
        for i, size in enumerate(comm_size):
            lmbd_mf_expanded[:, start_idx:start_idx + size] = np.tile(lmbd_mf[:, i].reshape(-1, 1), size)
            start_idx += size

        # Simulate combined process multiple times
        lmbd_comb_list = []
        X_comb_list = []
        lmbd_mf_list = []
        X_mf_list = []

        i=0
        for i in range(100):
            np.random.seed(graph_idx * 1000 + i)
            E = np.random.exponential(scale=1.0, size=(final_time, num_nodes, 15))
            _, lmbd_comb_new, X_comb_new = simulate_comb_sbm(E, final_time, num_nodes, comm_size, kernel_function_matrix, kernel_params_matrix, G, labs)
            _, lmbd_mf_new, X_mf_new = simulate_mf_sbm(E, final_time, comm_size, prob_matrix, kernel_function_matrix, kernel_params_matrix)
            final_time_lmbd_comb_new = lmbd_comb_new[-1, :]
            final_time_X_comb_new = X_comb_new[-1, :]
            final_time_lmbd_mf_new = lmbd_mf_new[-1,:]
            final_time_X_lmbd_mf_new = X_mf_new[-1,:]
            lmbd_comb_list.append(final_time_lmbd_comb_new)
            X_comb_list.append(final_time_X_comb_new)
            lmbd_mf_list.append(final_time_lmbd_mf_new)
            X_mf_list.append(final_time_X_lmbd_mf_new)
            i+= 1

        lmbd_mean_comb= np.mean(np.stack(lmbd_comb_list) - lmbd_aux[-1, :], axis=0)
        lmbd_comb_arr = np.stack(lmbd_comb_list) 
        X_comb_arr = np.stack(X_comb_list)   
        lmbd_mf_arr = np.stack(lmbd_mf_new)
        X_mf_arr = np.stack(X_mf_list)

        X_diff_mf = np.abs(X_comb_arr - X_mf_arr)    
        lmbd_diff = np.abs(lmbd_comb_arr - lmbd_mf_expanded[-1,:]) 

        # Compute metrics
        two_norm_comb = np.linalg.norm(lmbd_mean_comb, ord=2) / final_time
        infinity_norm_comb = np.linalg.norm(lmbd_mean_comb, ord=np.inf)
        two_norm_aux = np.linalg.norm(lmbd_aux[-1, :] - lmbd_mf_expanded[-1, :], ord=2) / final_time
        infinity_norm_aux = np.linalg.norm(lmbd_aux[-1, :] - lmbd_mf_expanded[-1, :], ord=np.inf)

        mean_dist_lmbd = np.mean(np.mean(lmbd_diff, axis = 1))
        mean_dist_X = np.mean(np.mean(X_diff_mf, axis = 1))
        max_dist_X = np.max(np.mean(X_diff_mf, axis = 1))
        max_dist_lmbd = np.max(np.mean(lmbd_diff, axis = 1))

        # Store for this graph realization
        two_norms_comb_all[graph_idx].append(two_norm_comb)
        infinity_norms_comb_all[graph_idx].append(infinity_norm_comb)
        two_norms_aux_all[graph_idx].append(two_norm_aux)
        infinity_norms_aux_all[graph_idx].append(infinity_norm_aux)
        mean_distances_lmbd_all[graph_idx].append(mean_dist_lmbd)
        mean_distances_X_all[graph_idx].append(mean_dist_X)
        max_distances_X_all[graph_idx].append(max_dist_X)
        max_distances_lmbd_all[graph_idx].append(max_dist_lmbd)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

results_dir = f"results/T_dep_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

np.save(f"{results_dir}/two_norms_comb.npy", two_norms_comb_all)
np.save(f"{results_dir}/infty_norms_comb.npy", infinity_norms_comb_all)
np.save(f"{results_dir}/two_norms_aux.npy", two_norms_aux_all)
np.save(f"{results_dir}/infty_norms_aux.npy", infinity_norms_aux_all)
np.save(f"{results_dir}/mean_distances.npy", mean_distances_lmbd_all)
np.save(f"{results_dir}/mean_distances_X.npy", mean_distances_X_all)
np.save(f"{results_dir}/max_distances.npy", max_distances_lmbd_all)
np.save(f"{results_dir}/max_distances_X.npy", max_distances_X_all)






# print("two_norms_comb shape:", np.shape(two_norms_comb))
# print("infinity_norms_comb shape:", np.shape(infinity_norms_comb))
# print("two_norms_aux shape:", np.shape(two_norms_aux))
# print("infinity_norms_aux shape:", np.shape(infinity_norms_aux))
# print("mean_distances shape:", np.shape(mean_distances_lmbd))
# print("mean_distances_X shape:", np.shape(mean_distances_X))

#plot_N(range_values, two_norms_comb, infinity_norms_comb, results_dir)
#plot_N(range_values, two_norms_aux, infinity_norms_aux, results_dir)


