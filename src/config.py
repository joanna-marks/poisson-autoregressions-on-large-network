import numpy as np
from helper_functions import gaussian_kernel, power_law_kernel, exponential_kernel

final_time = 400
num_nodes1 = 1000
num_nodes2 = 10000
alphas =  np.array([0.5,0.3,0.2]) 
comm_size1 = (num_nodes1 * alphas).astype(int)
comm_size2 = (num_nodes2 * alphas).astype(int)

prob_matrix = np.array([[0.9, 0.3, 0.1],
                               [0.3, 0.7, 0.2],
                               [0.1, 0.2, 0.6]])

#kernel_function = power_law_kernel
power_params = [0.4,5]
exp_params = [0.9,0.5]
gaussian_params = [0.9,0.5]


kernel_params_matrix = np.array([gaussian_params, power_params, exp_params])
kernel_function_matrix = np.array([gaussian_kernel, power_law_kernel, exponential_kernel])
# kernel_function_matrix = np.array([[gaussian_kernel, power_law_kernel, exponential_kernel],
#                                     [power_law_kernel, gaussian_kernel, exponential_kernel],
#                                     [power_law_kernel, exponential_kernel, gaussian_kernel]])

# kernel_params_matrix = np.array([[gaussian_params, power_params, exp_params],
#                                   [power_params, gaussian_params, exp_params],
#                                   [power_params, exp_params, gaussian_params]])





