import numpy as np
import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.ploting import plot_lmbdas_3group
from src.config import alphas, num_nodes1, num_nodes2
from src.saving import load_results


results_dir = "results/1000_10000_20250515_221653_20250515_221653"
lmbd_comb = np.load(f"{results_dir}/lmbd_1000.npy")
lmbd_comb_exp = np.load(f"{results_dir}/lmbd_10000.npy")
lmbd_mf = np.load(f"{results_dir}/lmbd_mf_1000.npy")

print(lmbd_mf[:, 0])

plot_lmbdas_3group(lmbd_comb, lmbd_comb_exp, lmbd_mf, alphas, num_nodes1, num_nodes2, results_dir, 'lambdas.pdf')