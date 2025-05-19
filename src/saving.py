import numpy as np
import os
from datetime import datetime

def save_results(lmbd_comb, lmbd_comb_exp, lmbd_mf,T, comm_size1, comm_size2, prob_matrix, kernel_function, kernel_parameters, experiment_name=None):

    # Create a timestamp for unique file identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create experiment subdirectory
    if experiment_name:
        exp_dir = f"{results_dir}/{experiment_name}_{timestamp}"
    else:
        exp_dir = f"{results_dir}/run_{timestamp}"
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    # Save individual arrays using NumPy's efficient binary format
    np.save(f"{exp_dir}/lmbd_1000.npy", lmbd_comb)
    np.save(f"{exp_dir}/lmbd_10000.npy", lmbd_comb_exp)
    np.save(f"{exp_dir}/lmbd_mf_1000.npy", lmbd_mf)

    # Save experiment metadata
    metadata = {
        'timestamp': timestamp,
        'final time': T,
        'num_nodes1': np.sum(comm_size1),
        'num_nodes2': np.sum(comm_size2),
        'alphas': comm_size1 / np.sum(comm_size1),
        'num_communities': len(comm_size1),
        'kernel_function': kernel_function,
        'kernel_parameters': kernel_parameters,
        'prob_matrix': prob_matrix.tolist(), 
    }
    
    # Save metadata as NumPy array for consistency
    np.save(f"{exp_dir}/metadata.npy", np.array(list(metadata.items()), dtype=object))
    
    print(f"Results saved to directory: {exp_dir}")
    return exp_dir


def load_results(results_dir):
    # Load lambda arrays
    lmbd_comb = np.load(f"{results_dir}/lmbd_1000.npy")
    lmbd_comb_exp = np.load(f"{results_dir}/lmbd_10000.npy")
    lmbd_mf = np.load(f"{results_dir}/lmbd_mf_1000.npy")
    
    # Load metadata if it exists
    try:
        metadata_array = np.load(f"{results_dir}/metadata.npy", allow_pickle=True)
        metadata = dict(metadata_array)
    except:
        metadata = {}
    
    return {
        "lmbd_comb": lmbd_comb,
        "lmbd_comb_exp": lmbd_comb_exp,
        "lmbd_mf": lmbd_mf,
        "metadata": metadata
    } 

