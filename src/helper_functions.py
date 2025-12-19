import numpy as np

# Define excitation function psi using a modified log-sum-exp function with clipping to avoid overflow
# def psi(x):
#     return np.clip(np.log(1 + np.exp(x)), None, 40)

def psi(x):
    return x

#Redefine poisson_randomness function to take E as an argument
def poisson_randomness(lmbd, E):
    i = 0
    S = 0  
    while S <= lmbd:  
        i += 1
        S += E[i]  
    return i - 1 

import numpy as np

def poisson_randomness_vectorized(lmbd_vec, E_vec):
    """
    Vectorized Poisson sampling using inverse transform method.
    
    Parameters:
    - lmbd_vec: shape (N,) vector of Poisson means (one per node)
    - E_vec: shape (N, M) array of pre-sampled exponential(1) variables per node
    
    Returns:
    - samples: shape (N,) array of Poisson samples
    """
    N, M = E_vec.shape  # N = num nodes, M = max number of events considered
    cum_sums = np.cumsum(E_vec, axis=1)  # Shape: (N, M)
    
    # For each row, find the first index where cumulative sum exceeds lmbd
    exceeds = cum_sums > lmbd_vec[:, None]  # Shape: (N, M), boolean
    first_exceed = np.argmax(exceeds, axis=1)  # First True along axis 1
    
    # If lmbd is very large and E_vec is too short, we may never exceed — handle edge case
    never_exceed = ~np.any(exceeds, axis=1)
    first_exceed[never_exceed] = M  # Set to max count if never exceeded

    return first_exceed - 1  # Same logic as original


    # Define different types of kernels and return their values at times 1 to some chosen values t
def exponential_kernel(kernel_params, t, lag=None):
    alpha, beta = kernel_params
    result = alpha * np.exp(-beta * t)
    if lag is not None:
        result = result * (t <= lag)
    return result

def power_law_kernel(kernel_params, t, lag=None):
    alpha, beta = kernel_params
    result = alpha / (1 + t ** beta)
    if  lag is not None:
        result = result * (t <= lag)
    return result

def rayleigh_kernel(kernel_params, t, lag=None):
    alpha, beta = kernel_params
    result = alpha * t * np.exp(-beta * t**2)
    if lag is not None:
        result = result * (t <= lag)
    return result

def gaussian_kernel(kernel_params, t, lag=None):
    alpha, beta = kernel_params
    result = alpha * np.exp(-0.5 * ((t - beta) ** 2))
    if lag is not None:
        result = result * (t <= lag)
    return result

def sigmoid_kernel(kernel_params, t, lag=None):
    alpha, beta = kernel_params
    result = np.tanh(alpha * t + beta)
    if lag is not None:
        result = result * (t <= lag)
    return result


#Helper functions for inference

def dirac_delta(x,y):
    " Dirac delta function "
    if x == y:
        return 1
    else:
        return 0


def plug_in_prob_matrix(adj_matrix, labs, comm_sizes):
    " Compute plug-in estimate of the probability matrix from adjacency matrix and community labels"
    A = np.asarray(adj_matrix)
    labs = np.asarray(labs)
    K = len(comm_sizes)

    # initialize plug-in probability matrix
    prob_matrix = np.zeros((K, K), dtype=float)

    # loop over all community pairs (ℓ, h)
    for l in range(K):
        nodes_l = np.where(labs == l)[0]
        for h in range(K):
            nodes_h = np.where(labs == h)[0]

            # sum over all edges j→i with i∈l, j∈h
            total_edges = 0.0
            for i in nodes_l:
                for j in nodes_h:
                    total_edges += A[i, j]

            # divide by product of community sizes
            if comm_sizes[l] > 0 and comm_sizes[h] > 0:
                prob_matrix[l, h] = total_edges / (comm_sizes[l] * comm_sizes[h])
            
            
            else:
                prob_matrix[l, h] = 0.0
    return prob_matrix


def padding(phi, T):
    p, _, _ = phi.shape
    pad_width = ((0, T - p), (0, 0), (0, 0), )   # pad only last axis
    if T > p:
        phi = np.pad(phi, pad_width, mode="constant", constant_values=0)
    return phi


def phi_tensor_to_param_vector(phi, p):
    """
    Convert phi of shape (T, num_comm, num_comm)
    to a flat vector of length num_comm * num_comm * p
    following the parameter ordering:
        (s, r, m) with s outer, r middle, m inner loop.
    """
    T, num_comm, _ = phi.shape
    assert T >= p, "phi must have at least p time steps at the front"

    phi_vec = []

    # Loop matching gradient index order:
    #   (s * num_comm + r) * p + (m - 1)
    for s in range(num_comm):       # destination
        for r in range(num_comm):   # source
            for m in range(1, p + 1):
                phi_vec.append(phi[m - 1, s, r])

    return np.array(phi_vec)

def phi_param_vector_to_tensor(phi_vec, num_comm, p):
    """
    Convert a flat phi parameter vector back to a phi tensor
    of shape (T, num_comm, num_comm).

    Ordering of phi_vec must be:
        (s = 0..num_comm-1)
            (r = 0..num_comm-1)
                (m = 1..p)

    Output tensor indexing:
        phi_tensor[t, s, r] = phi^{sr}_{t+1}
    with zero padding for t >= p.
    """
    expected_size = num_comm * num_comm * p
    assert len(phi_vec) == expected_size, "phi_vec length does not match num_comm*num_comm*p"

    phi_tensor = np.zeros((p, num_comm, num_comm))

    idx = 0
    for s in range(num_comm):           # destination
        for r in range(num_comm):       # source
            for m in range(1, p + 1):   # lag
                phi_tensor[m - 1, s, r] = phi_vec[idx]
                idx += 1

    return phi_tensor





# import matplotlib.pyplot as plt

# # # Time grid
# t = np.linspace(0.01, 10, 500)

# power_params =  [0.9, 2.5]
# ray_params = [0.5, 1.0]
# exp_params = [0.9, 0.5]

# # # Compute kernel values
# exp_vals = exponential_kernel(exp_params, t)
# rayleigh_vals = rayleigh_kernel(ray_params, t)
# power_vals = power_law_kernel(power_params, t)
# # Assuming exponential_kernel and t are already defined

# # First plot: Varying alpha
# exp_params_alpha1 = [1.5, 0.5]
# exp_params_alpha2 = [1.0, 0.5]
# exp_params_alpha3 = [0.5, 0.5]

# exp_vals_alpha1 = exponential_kernel(exp_params_alpha1, t)
# exp_vals_alpha2 = exponential_kernel(exp_params_alpha2, t)
# exp_vals_alpha3 = exponential_kernel(exp_params_alpha3, t)

# # Second plot: Varying beta
# exp_params_beta1 = [1.0, 1.0]
# exp_params_beta2 = [1.0, 0.5]
# exp_params_beta3 = [1.0, 0.2]

# exp_vals_beta1 = exponential_kernel(exp_params_beta1, t)
# exp_vals_beta2 = exponential_kernel(exp_params_beta2, t)
# exp_vals_beta3 = exponential_kernel(exp_params_beta3, t)

# # # Create side-by-side plots
# # plt.figure(figsize=(14, 5))

# # # Left plot: Varying alpha
# # plt.subplot(1, 2, 1)
# # plt.plot(t, exp_vals_alpha1, label=r"$\alpha$ = 1.5")
# # plt.plot(t, exp_vals_alpha2, label=r"$\alpha$ = 1.0")
# # plt.plot(t, exp_vals_alpha3, label=r"$\alpha$ = 0.5")
# # plt.title(r"Varying $\alpha$, Fixed $\beta = 0.5$")
# # plt.xlabel("Time")
# # plt.ylabel("Value")
# # plt.legend()
# # plt.grid(True)

# # # Right plot: Varying beta
# # plt.subplot(1, 2, 2)
# # plt.plot(t, exp_vals_beta1, label=r"$\beta$ = 1.0")
# # plt.plot(t, exp_vals_beta2, label=r"$\beta$ = 0.5")
# # plt.plot(t, exp_vals_beta3, label=r"$\beta$ = 0.2")
# # plt.title(r"Varying $\beta$, Fixed $\alpha = 1.0$")
# # plt.xlabel("Time")
# # plt.ylabel("Value")
# # plt.legend()
# # plt.grid(True)

# # plt.tight_layout()
# # plt.savefig("exp_kernels_alpha_beta_variation.pdf")
# # plt.show()

# # Time grid
# t = np.linspace(0.01, 10, 500)

# # Kernel parameters for exponential, Rayleigh, and Power-law
# power_params = [0.9, 2.5]
# ray_params = [0.5, 1.0]
# exp_params_1 = [1.5, 0.5]
# exp_params_2 = [1.0, 0.5]
# exp_params_3 = [1.0, 1.0]


# # Compute kernel values for each case
# exp_vals_1 = exponential_kernel(exp_params_1, t)
# exp_vals_2 = exponential_kernel(exp_params_2, t)
# exp_vals_3 = exponential_kernel(exp_params_3, t)


# rayleigh_vals = rayleigh_kernel(ray_params, t)
# power_vals = power_law_kernel(power_params, t)

# # Create side-by-side plots
# plt.figure(figsize=(18, 5))

# # Plot 1: Exponential kernel with different parameter combinations
# plt.subplot(1, 3, 1)
# plt.plot(t, exp_vals_1, label=r"$\alpha = 1.5, \beta = 0.5$", color='blue')
# plt.plot(t, exp_vals_2, label=r"$\alpha = 1.0, \beta = 0.5$", color='green')
# plt.plot(t, exp_vals_3, label=r"$\alpha = 1.0, \beta = 1.0$", color='red')
# plt.title(r"Exponential Kernel with Varying $\alpha$ and $\beta$")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)

# # Plot 2: Rayleigh kernel
# plt.subplot(1, 3, 2)
# plt.plot(t, rayleigh_vals, label="Rayleigh Kernel", color='blue')
# plt.title(r"Rayleigh Kernel ($\sigma = 0.5, \beta = 1.0$)")
# plt.xlabel("Time")
# plt.ylabel("Value")
# #plt.legend()
# plt.grid(True)

# # Plot 3: Power-law kernel
# plt.subplot(1, 3, 3)
# plt.plot(t, power_vals, label="Power-law Kernel", color='blue')
# plt.title(r"Power-law Kernel ($\gamma = 0.9, \delta = 2.5$)")
# plt.xlabel("Time")
# plt.ylabel("Value")
# #plt.legend()
# plt.grid(True)

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig("kernels_comparison_with_exponential_variations.pdf")
# plt.show()
