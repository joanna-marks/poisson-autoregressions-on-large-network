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


def expand_adjacency_matrix(labs, adj_matrix, comm_sizes_before, comm_sizes_after, prob_matrix):
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
    expanded_matrix[:old_nodes, :old_nodes] = adj_matrix
    
    additional_nodes = np.array(comm_sizes_after) - np.array(comm_sizes_before)

   # print("Additional nodes:", additional_nodes)
    new_comm_labels = []
    for comm_idx, size in enumerate(additional_nodes):
        new_comm_labels.extend([comm_idx] * size)

    #print("New community labels:", new_comm_labels)
    new_comm_labels = labs + new_comm_labels
    
   # print("New community labels:", new_comm_labels)
    
    for i in range(old_nodes, new_nodes):
        for j in range(new_nodes):
            prob = prob_matrix[new_comm_labels[i], new_comm_labels[j]]
            #print(prob)
            if np.random.rand() < prob:
                expanded_matrix[i, j] = 1

    for j in range(old_nodes, new_nodes):
        for i in range(old_nodes):
            prob = prob_matrix[new_comm_labels[i], new_comm_labels[j]]
            if np.random.rand() < prob:
                expanded_matrix[i, j] = 1

    sorted_indices = sorted(range(len(new_comm_labels)), key=lambda i: new_comm_labels[i])
    #print(sorted_indices)
    
    # 3. Create a mapping from old index to new index
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    #print("Old to new mapping:", old_to_new)    
    #print(old_to_new[3])

    # 4. Reorder the adjacency matrix
    reordered_matrix = np.zeros((new_nodes, new_nodes))
    for i in range(new_nodes):
        for j in range(new_nodes):
            reordered_matrix[old_to_new[i], old_to_new[j]] = expanded_matrix[i, j]
    #print("Reordered matrix:", reordered_matrix)
    #print("Reordered matrix:", reordered_matrix)
    
    # 5. Reorder the community labels
    reordered_labels = [new_comm_labels[i] for i in sorted_indices]

    
    return reordered_labels, reordered_matrix

# labs, G = generate_sbm_adjacency_matrix([3, 5, 5], np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]))
# labs_exp, G_exp = expand_adjacency_matrix(labs, G, [3, 5, 5], [30, 50, 50], np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]))
# labs_more, G_more  = generate_sbm_adjacency_matrix([4, 4, 10], np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]))
# print(G)
# print(G_exp)
# print(G_more)
