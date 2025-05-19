import numpy as np
import matplotlib.pyplot as plt

from src.config import alphas
from src.saving import load_results

def plot_lmbdas_3group(lmbd_comb, lmbd_comb_more, lmbd_mf, alphas, num_nodes1, num_nodes2, results_dir, label):
    
    comm_size1 = (num_nodes1 * np.array(alphas)).astype(int)
    comm_size2 = (num_nodes2 * np.array(alphas)).astype(int)
    
    # Indices for different groups in the first network size
    group1_index = 0
    group2_index = comm_size1[0] + 1
    group3_index = comm_size1[0] + comm_size1[1] + 1
    
    # Indices for different groups in the larger network size
    group1_index2 = 0
    group2_index2 = comm_size2[0] + 1
    group3_index2 = comm_size2[0] + comm_size2[1] + 1
    
    # Create a figure with 3 subplots arranged horizontally
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharex=True)
    
    # Plot the data for Group 1 in the first subplot
    axs[0].plot(lmbd_mf[:, 0], '--', markersize=2, label='Group 1 Mean-Field', color='red')
    axs[0].plot(lmbd_comb[:, group1_index], 'o', markersize=2, label='Group 1 (n=1000)', color='blue')
    axs[0].plot(lmbd_comb_more[:, group1_index2], 'o', markersize=2, label='Group 1 (n=10000)', color='orange')
    axs[0].set_ylabel('Intensity', fontsize=14)
    axs[0].set_title(r'$C_1$', fontsize=16)
    axs[0].set_xlabel('Time', fontsize=14)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()
    
    # Plot the data for Group 2 in the second subplot
    axs[1].plot(lmbd_mf[:, 1], '--', markersize=2, label='Group 2 Mean-Field', color='black')
    axs[1].plot(lmbd_comb[:, group2_index], 'o', markersize=2, label='Group 2 (n=1000)', color='deepskyblue')
    axs[1].plot(lmbd_comb_more[:, group2_index2], 'o', markersize=2, label='Group 2 (n=10000)', color='goldenrod')
    axs[1].set_ylabel('Intensity', fontsize=14)
    axs[1].set_xlabel('Time', fontsize=14)
    axs[1].set_title(r'$C_2$', fontsize=16)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].legend()
    
    # Plot the data for Group 3 in the third subplot
    axs[2].plot(lmbd_mf[:, 2], '--', markersize=2, label='Group 3 Mean-Field', color='brown')
    axs[2].plot(lmbd_comb[:, group3_index], 'o', markersize=2, label='Group 3 (n=1000)', color='purple')
    axs[2].plot(lmbd_comb_more[:, group3_index2], 'o', markersize=2, label='Group 3 (n=10000)', color='green')
    axs[2].set_xlabel('Time', fontsize=14)
    axs[2].set_ylabel('Intensity', fontsize=14)
    axs[2].set_title(r'$C_3$', fontsize=16)
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[2].legend()
    
    # Add a main title for the entire figure
    fig.suptitle(f'SBM Simulation Results for Three Groups', fontsize=18)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save the figure
    plt.savefig(f"{results_dir}/{label}")
    
    # Show the plot
    plt.show()


def plot_N(range_values, two_norms, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range_values, two_norms, label='Two Norms')
    plt.xlabel('Range Values', fontsize=12)
    plt.ylabel('Martix norm', fontsize=12)
    plt.title('Difference between true lambda and mean-field lambda', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/N_values.pdf")
    plt.show()