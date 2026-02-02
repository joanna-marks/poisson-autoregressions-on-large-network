import numpy as np
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)


from src.ploting import  plot_two_norms, plot_infinity_norms, plot_mean_distances_X, plot_distances_grid
#results_dir = "results/N_dep_20250703_002537"
results_dir = "results/N_dep_20260201_131610"

two_norms_comb = np.load(f"{results_dir}/two_norms_comb.npy")
infinity_norms_comb = np.load(f"{results_dir}/infty_norms_comb.npy")
two_norms_aux = np.load(f"{results_dir}/two_norms_aux.npy")
infinity_norms_aux = np.load(f"{results_dir}/infty_norms_aux.npy")
mean_distances_lmbd = np.load(f"{results_dir}/mean_distances.npy")
mean_distances_X = np.load(f"{results_dir}/mean_distances_X.npy")
max_distances_X = np.load(f"{results_dir}/max_distances_X.npy")
max_distances_lmbd = np.load(f"{results_dir}/max_distances.npy")

max_iter = 5100
min_iter = 100
iter_by= 1000

range_values = np.arange(min_iter, max_iter, iter_by)

# # plot_N(
# #     range_values,
# #     two_norms_comb,
# #     two_norms_aux,
# #     results_dir,
# #     'N_two_norms.pdf',
# #     name=r"Two Norm between $\lambda_T$ and $\bar{\lambda}_T$, $\hat{\lambda}_T$",
# #     norm_label='Two Norm'
# # )

# # plot_N(
# #     range_values,
# #     infinity_norms_comb,
# #     infinity_norms_aux,
# #     results_dir,
# #     'N_infty_norms.pdf',
# #     name=r"Infinity Norm between $\lambda_T$ and $\bar{\lambda}_T$, $\hat{\lambda}_T$",
# #     norm_label='Infinity Norm'
# # )


# print(two_norms_comb.dtype)
# print(two_norms_comb.shape)


# plot_all_norms_side_by_side(
#     range_values,
#     two_norms_comb,
#     two_norms_aux,
#     infinity_norms_comb,
#     infinity_norms_aux,
#     results_dir,
#     save_name='all_norms.pdf'
# )

# plot_all_norms_with_multiple_lines(
#     range_values,
#     two_norms_comb,
#     two_norms_aux,
#     infinity_norms_comb,
#     infinity_norms_aux,
#     results_dir
# )

plot_mean_distances_X(
    range_values,
    mean_distances_X,
    results_dir
)

plot_mean_distances_X(
    range_values,
    mean_distances_lmbd,
    results_dir,
    save_name='mean_distances_lmbd.pdf'
)

plot_mean_distances_X(
    range_values,
    mean_distances_lmbd,
    results_dir,
    save_name='mean_distances_lmbd.pdf'
)


plot_distances_grid(range_values, mean_distances_X, mean_distances_lmbd, save_name="mean_distances_X_and_lmbd_log.pdf", results_dir=results_dir)
plot_distances_grid(range_values, mean_distances_X, mean_distances_lmbd, save_name="mean_distances_X_and_lmbd_lin.pdf", use_log = "linear", ref_line_slope=None, results_dir=results_dir)
plot_distances_grid(range_values, max_dist_X = max_distances_X, max_dist_lmbd = max_distances_lmbd, save_name="max_distances_X_and_lmbd.pdf", results_dir=results_dir, ref_line_slope=None )
plot_distances_grid(range_values, max_dist_X = max_distances_X, max_dist_lmbd = max_distances_lmbd, save_name="max_distances_X_and_lmbd_lin.pdf", use_log="linear", ref_line_slope=None, results_dir=results_dir)

plot_two_norms(range_values, two_norms_comb, two_norms_aux, results_dir,
               save_name="two_norms_linear.pdf", use_log="linear")

# Infinity-norms on log–log
plot_infinity_norms(range_values, infinity_norms_comb, infinity_norms_aux, results_dir,
                    save_name="inf_norms_loglog.pdf", use_log="loglog")








