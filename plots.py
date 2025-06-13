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


results_dir = "results/500_5000_20250613_105825_20250613_105825"
lmbd_comb = np.load(f"{results_dir}/lmbd_1000.npy")
lmbd_comb_exp = np.load(f"{results_dir}/lmbd_10000.npy")
lmbd_mf = np.load(f"{results_dir}/lmbd_mf_1000.npy")
lmbd_aux = np.load(f"{results_dir}/lmbd__aux_1000.npy")
lmbd_aux_exp = np.load(f"{results_dir}/lmbd_aux_10000.npy")


plot_lmbdas_3group(lmbd_comb, lmbd_comb_exp, lmbd_mf,lmbd_aux, lmbd_aux_exp, alphas, num_nodes1, num_nodes2, results_dir, 'lambdas.pdf')