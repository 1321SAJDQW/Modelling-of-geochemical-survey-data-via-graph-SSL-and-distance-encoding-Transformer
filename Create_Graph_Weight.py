# This code constructs a geochemical graph with edge weights.

import libpysal
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse as sp
import warnings

warnings.filterwarnings('ignore')

# Load data
inputfile = "H:/Data/Nantianshan.csv"
dataframe = pd.read_csv(inputfile)


xy = dataframe[['XX', 'YY']].values
features = dataframe.drop(columns=['FID', 'XX', 'YY', 'Label']).values

# Build graph with weighted edges
for distance in [1500]:
    wid = libpysal.weights.DistanceBand.from_array(xy, threshold=distance, p=2, binary=False, silence_warnings=True)
    sparse_adj = wid.sparse
    rows, cols = sparse_adj.nonzero()

    mask = rows < cols
    rows, cols = rows[mask], cols[mask]

    cos_sim_values = [max(cosine_similarity([features[i]], [features[j]])[0, 0], 0) for i, j in zip(rows, cols)]

    n_nodes = len(xy)
    data = cos_sim_values + cos_sim_values
    row_indices = list(rows) + list(cols)
    col_indices = list(cols) + list(rows)
    weights_sparse = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).tocsr()

    graph_data = {
        'adjacency_matrix': weights_sparse,
        'neighbors': wid.neighbors,
        'features': features,
        'coordinates': xy
    }

    with open(f"graph_{distance}_weighted.pkl", 'wb') as file:
        pkl.dump(graph_data, file)

    print(f"Graph with distance {distance} and weighted edges saved.")
    print(f"Number of edges: {len(rows)}")
    print(f"Average weight: {np.mean(cos_sim_values):.4f}")
