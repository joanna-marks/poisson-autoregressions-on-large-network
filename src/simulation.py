import numpy as np
from helper_functions import psi, poisson_randomness

#Simulating Hawkes with SBM structure
def simulate_comb_sbm(E, T, num_nodes, comm_size, kernel_function, kernel_params, G, mu_vector = np.array([0.2, 0.2, 0.2])):
    t_values = np.arange(T, 0, -1)
    kernel_values = np.zeros((T, num_nodes))
    for i in range(num_nodes):
        kernel_values[:, i] = kernel_function(kernel_params, t_values)

    if len(mu_vector) != comm_size.shape[0]:
        raise ValueError("The length of mu_vector must match the number of communities in comm_size.")

    mu_vector_expanded = np.zeros(num_nodes)
    start_idx = 0
    for i, size in enumerate(comm_size):
        mu_vector_expanded[start_idx:start_idx + size] = mu_vector[i]
        start_idx += size

    # Initializing the matrices
    N = np.zeros((T, num_nodes))
    lmbd = np.zeros((T, num_nodes))
    X = np.zeros((T, num_nodes))

    #Settting in itial values
    lmbd[0, :] = np.full(num_nodes, psi(mu_vector[0]))
    X[0, :] = np.zeros(num_nodes)
    
    for i in range(1, T):
        for j in range(num_nodes):
            lmbd[i, j] = psi(mu_vector_expanded[j] + np.dot(G[j, :], np.sum(kernel_values[T-i:, :] * X[:i, :], axis=0)) / num_nodes)
            X[i, j] = poisson_randomness(lmbd[i, j],E[i,j, :])
            N[i, j] = N[i-1, j] + X[i, j]
            N[i, j] = N[i-1, j] + X[i, j]

    return N, lmbd, X


def simulate_mf_sbm(E, T, comm_size, prob_matrix, kernel_function, kernel_params, mu_vector = np.array([0.2,0.2, 0.2])):
    num_comm = len(comm_size)
    alphas = comm_size/np.sum(comm_size)
    t_values = np.arange(T,0, -1)
    
    kernel_values = np.zeros((T, num_comm))
    for i in range(num_comm):
        kernel_values[:, i] = kernel_function(kernel_params, t_values)


    # Initialize vectors to store values for the simulation
    N = np.zeros((T, num_comm))
    lmbd = np.zeros((T, num_comm))
    X = np.zeros((T, num_comm))   

    # Set initial values for lambda and counts
    X[0, :] = np.zeros(num_comm)
    N[0, :] = X[0, :]
    lmbd[0, :] = psi(mu_vector)

    # Run the simulation for each time step
    for i in range(1, T):
        for j in range(num_comm):
            lmbd[i, j] = psi(mu_vector[j] + np.dot(alphas* prob_matrix[:,j], np.sum(kernel_values[T-i:,:] * lmbd[:i,:], axis = 0)))

            # Simulate the Poisson process for the current time step
            X[i] = poisson_randomness(lmbd[i,j],  E[i, j, :])
        
            # Update the cumulative count of events
            N[i] = N[i - 1] + X[i]
    
    print(i)

    return N, lmbd, X
