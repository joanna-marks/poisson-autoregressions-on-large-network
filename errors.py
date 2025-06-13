import numpy as np
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.ploting import plot_lmbdas_3group
from src.config import alphas, num_nodes1, num_nodes2, comm_size1, comm_size2, final_time
from src.saving import load_results



results_dir = "results/1000_10000_20250531_141256_20250531_141256"
lmbd_comb = np.load(f"{results_dir}/lmbd_1000.npy")
lmbd_comb_exp = np.load(f"{results_dir}/lmbd_10000.npy")
lmbd_mf = np.load(f"{results_dir}/lmbd_mf_1000.npy")
lmbd_aux = np.load(f"{results_dir}/lmbd__aux_1000.npy")
lmbd_aux_exp = np.load(f"{results_dir}/lmbd_aux_10000.npy")


# Compute error per community for each method
def compute_error_per_community(lmbd, lmbd_mf, comm_sizes):
    start = 0
    error_per_comm = []
    for size in comm_sizes:
        end = start + size
        error_per_comm.append(np.mean(lmbd[:, start:end] - lmbd_mf[:, start:end]))
        start = end
    return error_per_comm

start_idx = 0
lmbd_mf = np.zeros_like(lmbd_comb)
for i, size in enumerate(comm_size1):
        lmbd_mf[:, start_idx:start_idx + size] = np.tile(lmbd_mf[:, i].reshape(-1, 1), size)
        start_idx += size

start_idx = 0
lmbd_mf_exp = np.zeros_like(lmbd_comb_exp)
for i, size in enumerate(comm_size2):
        lmbd_mf_exp[:, start_idx:start_idx + size] = np.tile(lmbd_mf_exp[:, i].reshape(-1, 1), size)
        start_idx += size
        

error_comb = np.abs(lmbd_comb[:,0] - lmbd_mf[:,0]).mean()
error_comb_exp = np.abs(lmbd_comb_exp[:,0] - lmbd_mf_exp[:,0]).mean()
#error_comb_exp = ((lmbd_comb_exp[:,0] - lmbd_mf_exp[:,0])**2).mean()
error_aux = np.mean(np.abs(lmbd_aux - lmbd_mf))
error_aux_exp = np.mean(np.abs(lmbd_aux_exp - lmbd_mf_exp))

error_comb_per_comm = compute_error_per_community(lmbd_comb, lmbd_mf, comm_size1)
error_comb_exp_per_comm = compute_error_per_community(lmbd_comb_exp, lmbd_mf_exp, comm_size2)
error_aux_per_comm = compute_error_per_community(lmbd_aux, lmbd_mf, comm_size1)
error_aux_exp_per_comm = compute_error_per_community(lmbd_aux_exp, lmbd_mf_exp, comm_size2)


print("Error (Combined):", error_comb)
print("Error (Combined Exp):", error_comb_exp)
print("Error (Aux):", error_aux)
print("Error (Aux Exp):", error_aux_exp)
print("Error per community (Combined):", error_comb_per_comm)
print("Error per community (Combined Exp):", error_comb_exp_per_comm)
print("Error per community (Aux):", error_aux_per_comm)
print("Error per community (Aux Exp):", error_aux_exp_per_comm)
