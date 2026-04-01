import libpysal
import pandas as pd
import pickle as pkl
import numpy as np
from scipy import sparse as sp
import warnings

warnings.filterwarnings('ignore')

inputfile = "data.csv"
dataframe = pd.read_csv(inputfile)

xy = dataframe[['XX', 'YY']].values
features = dataframe.drop(columns=['FID', 'XX', 'YY', 'Label']).values

for distance in [1500]:
    wid = libpysal.weights.DistanceBand.from_array(xy, threshold=distance, p=2, binary=False, silence_warnings=True)
    sparse_adj = wid.sparse
    rows, cols = sparse_adj.nonzero()

    mask = rows < cols
    rows, cols = rows[mask], cols[mask]

    norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    norm_features[np.isnan(norm_features)] = 0.0

    sim_values = np.sum(norm_features[rows] * norm_features[cols], axis=1)
    cos_sim_values = np.clip(sim_values, 0, None)

    n_nodes = len(xy)
    data = np.concatenate([cos_sim_values, cos_sim_values])
    row_indices = np.concatenate([rows, cols])
    col_indices = np.concatenate([cols, rows])

    weights_sparse = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes)).tocsr()

    graph_data = {
        'adjacency_matrix': weights_sparse,
        'neighbors': wid.neighbors,
        'features': features,
        'coordinates': xy
    }

    with open(f"graph_{distance}_weighted.pkl", 'wb') as file:
        pkl.dump(graph_data, file)

    print(f"Graph distance: {distance}")
    print(f"Edges: {len(rows)}")
    print(f"Average weight: {np.mean(cos_sim_values):.4f}")