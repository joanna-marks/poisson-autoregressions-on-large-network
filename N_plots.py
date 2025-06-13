import numpy as np
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)


from src.ploting import plot_N
results_dir = "results/N_dep_20250605_153740"

two_norms_comb = np.load(f"{results_dir}/two_norms_comb.npy")
infinity_norms_comb = np.load(f"{results_dir}/infty_norms_comb.npy")
two_norms_aux = np.load(f"{results_dir}/two_norms_aux.npy")
infinity_norms_aux = np.load(f"{results_dir}/infty_norms_aux.npy")
mean_distances_lmbd = np.load(f"{results_dir}/mean_distances.npy")
mean_distances_X = np.load(f"{results_dir}/mean_distances_X.npy")

max_iter = 1000
min_iter = 100
iter_by= 50

range_values = np.arange(min_iter, max_iter, iter_by)

plot_N(range_values, two_norms_comb, infinity_norms_comb, results_dir, 'N_true.pdf',name= r"Matrix norms between $\bar{\lambda}_T$ and $\lambda_T$" )
plot_N(range_values, two_norms_aux, infinity_norms_aux, results_dir, 'N_bar.pdf' , name= r"Matrix norms between $\hat{\lambda}_T$ and $\bar{\lambda}_T$")
plot_N(range_values, mean_distances_lmbd, mean_distances_X, results_dir, 'N_mean.pdf', name= r"Mean distances between $\bar{\lambda}_T$ and $\lambda_T$ and between $\hat{\lambda}_T$ and $\bar{\lambda}_T$")


