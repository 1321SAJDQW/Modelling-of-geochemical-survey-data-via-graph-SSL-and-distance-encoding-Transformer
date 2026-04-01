import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

# ----------------------------------------------------------------------------------------------------------------------
# Load Graph Data and Extract Coordinates and Adjacency Matrix
# ----------------------------------------------------------------------------------------------------------------------
with open('graph_1500_weighted.pkl', 'rb') as rf1:
    graph = pkl.load(rf1)

xy = graph['coordinates']
n_nodes = len(xy)
weights_sparse = graph['adjacency_matrix']

unweighted_adj = weights_sparse.copy()
unweighted_adj[unweighted_adj > 0] = 1

shortest_paths_float = shortest_path(csgraph=unweighted_adj, directed=False, unweighted=True)
shortest_paths_float[np.isinf(shortest_paths_float)] = 0
shortest_paths = shortest_paths_float.astype(int)

np.fill_diagonal(shortest_paths, 1)
shortest_paths[shortest_paths > 3] = 0

print("Shortest path matrix shape:", shortest_paths.shape)
print("Shortest path matrix (top left 15x15):")
print(shortest_paths[:15, :15])

np.save('shortest_paths_matrix.npy', shortest_paths)