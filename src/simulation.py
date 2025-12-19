import numpy as np
from helper_functions import psi, poisson_randomness, poisson_randomness_vectorized
from tqdm import tqdm
from scipy.sparse import csr_matrix


#Simulating Hawkes with SBM structure
def simulate_comb_sbm(E, T, num_nodes, comm_size, kernel_function_matrix, kernel_params_matrix, G, labs, mu_vector = np.array([0.5, 0.5, 0.5]), lag = None):
    num_comm = len(comm_size)
    t_values = np.arange(T, 0, -1)

    #Compute a matrix of kernel_valuesa at times 0:T for all nodes
    kernel_values = np.zeros((T, num_nodes))
    for i in range(num_nodes):
       # for j in range(num_nodes):
        kernel_function = kernel_function_matrix[labs[i]]
        kernel_params = kernel_params_matrix[labs[i]]
        kernel_values[:, i] = kernel_function(kernel_params, t_values, lag = lag) 

    if len(mu_vector) != comm_size.shape[0]:
        raise ValueError("The length of mu_vector must match the number of communities in comm_size.")

    mu_vector_expanded = np.zeros(num_nodes)
    mu_vector_expanded = mu_vector[labs]

    # Initializing the matrices
    N = np.zeros((T, num_nodes))
    lmbd = np.zeros((T, num_nodes))
    X = np.zeros((T, num_nodes))

    #Settting initial values for lambda and counts
    lmbd[0, :] = psi(mu_vector_expanded)
    X[0, :] = np.zeros(num_nodes)
    
    for i in range(1, T):
        for j in range(num_nodes):
            lmbd[i, j] = psi(mu_vector_expanded[j] + np.dot(G[j, :], np.sum(kernel_values[T-i:, :] * X[:i, :], axis=0)) / num_nodes)
            X[i, j] = poisson_randomness(lmbd[i, j],E[i,j, :])
            N[i, j] = N[i-1, j] + X[i, j]

    return N, lmbd, X

def simulate_comb_sbm_optimized(
    E, T, num_nodes, comm_size,
    kernel_function_matrix, kernel_params_matrix,
    G, labs,
    mu_vector=np.array([0.5, 0.2, 0.1])
):
    num_comm = len(comm_size)
    t_values = np.arange(T, 0, -1)

    if len(mu_vector) != num_comm:
        raise ValueError("The length of mu_vector must match the number of communities in comm_size.")

    # Precompute kernel values for all community pairs
    kernel_cache = {}
    for i in range(num_comm):
        for j in range(num_comm):
            func = kernel_function_matrix[i, j]
            params = kernel_params_matrix[i, j]
            kernel_cache[(i, j)] = func(params, t_values)

    # Precompute kernel values between all node pairs
    K_tensor = np.zeros((T, num_nodes, num_nodes))  # shape (T, i, j)
    for i in range(num_nodes):
        for j in range(num_nodes):
            c_i, c_j = labs[i], labs[j]
            K_tensor[:, i, j] = kernel_cache[(c_i, c_j)]

    # Expand mu to each node
    mu_vector_expanded = mu_vector[labs]

    # Initialize result arrays
    N = np.zeros((T, num_nodes))
    lmbd = np.zeros((T, num_nodes))
    X = np.zeros((T, num_nodes))

    # Initial values
    lmbd[0, :] = psi(mu_vector_expanded)

    # Convert G to sparse matrix for efficient multiplication
    G_sparse = csr_matrix(G)

    # Main time loop
    for t in tqdm(range(1, T)):
        kernel_effect = np.zeros(num_nodes)

        # Compute convolution-like effect from past activity using precomputed kernels
        for j in range(num_nodes):
            x_hist = X[:t, j]
            for i in range(num_nodes):
                kernel_vals = K_tensor[T - t:, i, j]  # Only relevant past kernel
                kernel_effect[i] += np.dot(kernel_vals, x_hist)

        # Update intensity
        lmbd[t, :] = psi(mu_vector_expanded + G_sparse.dot(kernel_effect) / num_nodes)

        # Sample new events
        X[t, :] = poisson_randomness_vectorized(lmbd[t, :], E[t, :, :])

        # Update cumulative count
        N[t, :] = N[t - 1, :] + X[t, :]

    return N, lmbd, X


def simulate_aux_sbm(E, T, num_nodes, comm_size, kernel_function_matrix, kernel_params_matrix, G, labs,  mu_vector = np.array([0.5, 0.5, 0.5])):
    num_comm = len(comm_size)
    t_values = np.arange(T, 0, -1)

    #Compute a matrix of kernel_valuesa at times 0:T for all nodes
    kernel_values = np.zeros((T, num_nodes))
    for i in range(num_nodes):
       # for j in range(num_nodes):
        kernel_function = kernel_function_matrix[labs[i]]
        kernel_params = kernel_params_matrix[labs[i]]
        kernel_values[:, i] = kernel_function(kernel_params, t_values)

    if len(mu_vector) != comm_size.shape[0]:
        raise ValueError("The length of mu_vector must match the number of communities in comm_size.")

    mu_vector_expanded = np.zeros(num_nodes)
    mu_vector_expanded = mu_vector[labs]

    # Initializing the matrices
    N = np.zeros((T, num_nodes))
    lmbd = np.zeros((T, num_nodes))
    X = np.zeros((T, num_nodes))

    #Settting initial values for lambda and counts
    lmbd[0, :] = psi(mu_vector_expanded)
    X[0, :] = np.zeros(num_nodes)
    
    for i in range(1, T):
        for j in range(num_nodes):
            lmbd[i, j] = psi(mu_vector_expanded[j] + np.dot(G[j, :], np.sum(kernel_values[T-i:, :] * lmbd[:i, :], axis=0)) / num_nodes)
            X[i, j] = poisson_randomness(lmbd[i, j],E[i,j, :])
            N[i, j] = N[i-1, j] + X[i, j]

    return N, lmbd, X



def simulate_mf_sbm(E, T, comm_size, prob_matrix, kernel_function_matrix, kernel_params_matrix, mu_vector = np.array([0.5, 0.5, 0.5]), lag = None):
    num_comm = len(comm_size)
    num_nodes = np.sum(comm_size)
    alphas = comm_size/np.sum(comm_size)
    t_values = np.arange(T,0, -1)
    
    kernel_values = np.zeros((T, num_comm))
    for i in range(num_comm):
        # for j in range(num_comm):
        kernel_function = kernel_function_matrix[i]
        kernel_params = kernel_params_matrix[i]
        kernel_values[:, i] = kernel_function(kernel_params, t_values, lag = lag)


    # Initialize vectors to store values for the simulation
    N = np.zeros((T, num_nodes))
    lmbd = np.zeros((T, num_comm))
    X = np.zeros((T, num_nodes))   

    # Set initial values for lambda and counts
    X[0, :] = np.zeros(num_nodes)
    N[0, :] = X[0, :]
    lmbd[0, :] = psi(mu_vector)

    # Run the simulation for each time step
    for i in range(1, T):
        for j in range(num_comm):
            lmbd[i, j] = psi(mu_vector[j] + np.dot(alphas* prob_matrix[:,j], np.sum(kernel_values[T-i:,:] * lmbd[:i,:], axis = 0)))
            size = comm_size[j]

            for k in range(size):
                # Simulate the Poisson process for the current time step
                X[i,k] = poisson_randomness(lmbd[i,j],  E[i, k, :])
            
                # Update the cumulative count of events
                N[i,k] = N[i - 1, k] + X[i, k]

    return N, lmbd, X, kernel_values
