import numpy as np
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.synthetic_data import generate_sbm_adjacency_matrix, expand_adjacency_matrix
from src.config import final_time, num_nodes1, num_nodes2, comm_size1, comm_size2, prob_matrix, kernel_function_matrix, kernel_params_matrix
from src.simulation import simulate_comb_sbm, simulate_mf_sbm
from src.saving import save_results



E = np.random.exponential(scale = 1.0, size = (final_time, num_nodes1, 20))
labs, G = generate_sbm_adjacency_matrix(comm_size1, prob_matrix)

N_comb, lmbd_comb ,X_comb = simulate_comb_sbm(E, final_time,num_nodes1,comm_size1, kernel_function_matrix, kernel_params_matrix, G, labs)
N_mf, lmbd_mf, X_mf = simulate_mf_sbm(E, final_time, comm_size1, prob_matrix, kernel_function_matrix, kernel_params_matrix)
print(f"Simulation for {num_nodes1} completed.")

E = np.random.exponential(scale = 1.0, size = (final_time, num_nodes2, 20))
#labs, G_exp = generate_sbm_adjacency_matrix(comm_size2, prob_matrix)
labs_exp, G_exp = expand_adjacency_matrix(labs, G, comm_size1, comm_size2, prob_matrix)

N_comb_exp, lmbd_comb_exp ,X_comb_exp = simulate_comb_sbm(E, final_time,num_nodes2,comm_size2, kernel_function_matrix, kernel_params_matrix, G_exp, labs_exp)
N_mf_exp, lmbd_mf_exp, X_mf_exp = simulate_mf_sbm(E, final_time, comm_size2, prob_matrix, kernel_function_matrix, kernel_params_matrix)
print(f"Simulation for {num_nodes2} completed.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = save_results(
    lmbd_comb, lmbd_comb_exp, lmbd_mf,
    comm_size1, comm_size2,
    final_time,
    prob_matrix,
    kernel_function=kernel_function_matrix,
    kernel_parameters=kernel_params_matrix,
    experiment_name=f"{num_nodes1}_{num_nodes2}_{timestamp}"
)