import numpy as np

final_time = 1000
num_nodes1 = 1000
num_nodes2 = 10000
alphas =  np.array([0.5,0.3,0.2]) 
comm_size1 = num_nodes1 * alphas
comm_size2 = num_nodes2 * alphas

prob_matrix = np.array([[0.9, 0.1, 0.1],
                               [0.1, 0.7, 0.8],
                               [0.1, 0.8, 0.3]])



