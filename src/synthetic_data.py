import numpy as np

def generate_sbm_adjacency_matrix(community_sizes, probability_matrix):
    np.random.seed(123)

    # Calculate the total number of nodes
    num_nodes = sum(community_sizes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    community_labels = [c for c, size in enumerate(community_sizes) for _ in range(size)]
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            prob = probability_matrix[community_labels[i], community_labels[j]]
            # Add an edge with probability `prob`
            if np.random.rand() < prob:
                adjacency_matrix[i, j] = 1

    return community_labels, adjacency_matrix


def expand_adjacency_matrix(adj_matrix, comm_sizes_before, comm_sizes_after, prob_matrix):
    np.random.seed(123)  
    
    # Verify input
    if len(comm_sizes_before) != len(comm_sizes_after):
        raise ValueError("Community count must be the same before and after expansion")
    
    for i in range(len(comm_sizes_before)):
        if comm_sizes_after[i] < comm_sizes_before[i]:
            raise ValueError("Communities can only expand, not shrink")
    
    # Calculate total nodes before and after
    old_nodes = sum(comm_sizes_before)
    new_nodes = sum(comm_sizes_after)
    
    if old_nodes != adj_matrix.shape[0]:
        raise ValueError("Input adjacency matrix size doesn't match comm_sizes_before")
    
    # Create the new adjacency matrix
    expanded_matrix = np.zeros((new_nodes, new_nodes))
    
    # Copy existing connections
    expanded_matrix[:old_nodes, :old_nodes] = adj_matrix
    
    # Create old community labels
    old_comm_labels = []
    for comm_idx, size in enumerate(comm_sizes_before):
        old_comm_labels.extend([comm_idx] * size)
    
    # Create new community labels
    new_comm_labels = []
    for comm_idx, size in enumerate(comm_sizes_after):
        new_comm_labels.extend([comm_idx] * size)
    
    # Track the starting index of each new community
    new_comm_start_indices = [0]
    running_sum = 0
    for size in comm_sizes_after[:-1]:
        running_sum += size
        new_comm_start_indices.append(running_sum)
    
    # Connect new nodes to existing nodes and among themselves
    old_node_count = 0
    for old_comm_idx, old_size in enumerate(comm_sizes_before):
        old_node_count += old_size
        new_size = comm_sizes_after[old_comm_idx]
        start_idx = new_comm_start_indices[old_comm_idx]
        
        # Only process new nodes in this community
        for i in range(start_idx + old_size, start_idx + new_size):
            # Connect to existing nodes in all communities
            for j in range(old_nodes):
                old_comm_j = old_comm_labels[j]
                prob = prob_matrix[old_comm_idx, old_comm_j]
                if np.random.rand() < prob:
                    expanded_matrix[i, j] = expanded_matrix[j, i] = 1
            
            # Connect to other new nodes
            for j in range(old_nodes, new_nodes):
                if i != j:  # Avoid self-loops
                    j_comm = new_comm_labels[j]
                    prob = prob_matrix[old_comm_idx, j_comm]
                    if np.random.rand() < prob:
                        expanded_matrix[i, j] = expanded_matrix[j, i] = 1
    
    return new_comm_labels, expanded_matrix
