import numpy as np
import os
import sys
from datetime import datetime


current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.synthetic_data import generate_sbm_adjacency_matrix
from src.simulation import simulate_comb_sbm, simulate_mf_sbm
from src.saving import save_results
from src.config_infernce import final_time, num_nodes2, comm_size2, prob_matrix, kernel_function_matrix, kernel_params_matrix, mu0, phi0, alphas, p, phi_true, mu_true
from src.inference import gradient_descent_fit
from src.ploting import plot_mu_estimates

labs, adj_matrix = generate_sbm_adjacency_matrix(comm_size2, prob_matrix)
E = np.random.exponential(scale = 1.0, size = (final_time, num_nodes2, 100))


N_comb, lmbd_comb ,X_comb ,kernel_values = simulate_comb_sbm(E, final_time,num_nodes2,comm_size2, kernel_function_matrix, kernel_params_matrix, adj_matrix, labs)
N_mf, lmbd_mf, X_mf, kernel_values = simulate_mf_sbm(E, final_time, comm_size2, prob_matrix, kernel_function_matrix, kernel_params_matrix)


mu_hat, phi_hat, theta_path, loglik_hist = gradient_descent_fit(
    mu0, phi0, alphas, prob_matrix, comm_size2,
    p, X_comb, labs,
    lr=1e-6, max_iter=1000
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = f"inference_results/{num_nodes2}_nodes_{timestamp}"

plot_mu_estimates(theta_path, mu_true, save_dir)

