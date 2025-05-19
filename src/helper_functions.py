import numpy as np

# Define excitation function psi using a modified log-sum-exp function with clipping to avoid overflow
def psi(x):
    return np.clip(np.log(1 + np.exp(x)), None, 40)

#Redefine poisson_randomness function to take E as an argument
def poisson_randomness(lmbd, E):
    i = 0
    S = 0  
    while S <= lmbd:  
        i += 1
        S += E[i]  
    return i - 1 

    # Define different types of kernels and return their values at times 1 to some chosen values t
def exponential_kernel(kernel_params, t):
    alpha, beta = kernel_params
    return alpha * np.exp(-beta * t)

def power_law_kernel(kernel_params, t):
    alpha, beta = kernel_params
    return alpha / (t ** (beta))

def rayleigh_kernel(kernel_params, t):
    alpha, beta = kernel_params
    return alpha * t * np.exp(-beta * t**2)

def gaussian_kernel(kernel_params, t):
    alpha, beta = kernel_params
    return alpha * np.exp(-0.5 * ((t - beta) ** 2))

def sigmoid_kernel(kernel_params, t):
    alpha, beta = kernel_params
    return np.tanh(alpha * t + beta)