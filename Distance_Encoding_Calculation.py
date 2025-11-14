import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp

# ----------------------------------------------------------------------------------------------------------------------
# Load Graph Data and Extract Coordinates and Adjacency Matrix
# ----------------------------------------------------------------------------------------------------------------------

with open('graph_1500_weighted.pkl', 'rb') as rf1:
    graph = pkl.load(rf1)

# Get node coordinates and adjacency matrix
xy = graph['coordinates']
n_nodes = len(xy)

weights_sparse = graph['adjacency_matrix']
adj_matrix = weights_sparse.toarray()

shortest_paths = np.zeros((n_nodes, n_nodes), dtype=int)

# ----------------------------------------------------------------------------------------------------------------------
# Calculate Shortest Paths Between Nodes
# ----------------------------------------------------------------------------------------------------------------------

for i in range(n_nodes):
    for j in range(n_nodes):
        if i == j:
            shortest_paths[i, j] = 1
        elif adj_matrix[i, j] > 0:
            shortest_paths[i, j] = 1
        else:
            shortest_paths[i, j] = abs(i - j)


shortest_paths[shortest_paths > 3] = 0

# ----------------------------------------------------------------------------------------------------------------------
# Output and Save Results
# ----------------------------------------------------------------------------------------------------------------------

print("Shortest path matrix shape:", shortest_paths.shape)
print("Adjacency matrix shape:", adj_matrix.shape)

print("Shortest path matrix (top left 15x15):")
print(shortest_paths[:15, :15])

np.save('shortest_paths_matrix.npy', shortest_paths)
