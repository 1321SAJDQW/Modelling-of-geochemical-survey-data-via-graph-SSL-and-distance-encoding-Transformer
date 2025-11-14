# This code is adapted from Zuo and Xu (2023).
# It constructs a geochemical graph without edge weights.

import libpysal
import pandas as pd
import pickle as pkl


# Read data
inputfile = "Data.csv"
dataframe = pd.read_csv(inputfile)

# read coordinates
xy = dataframe[['POINT_X', 'POINT_Y']]
xy = xy.values
for distance in [1500]:

    wid = libpysal.weights.distance.DistanceBand.from_array(xy, threshold=distance, p=2, binary=False)
    dict = wid.neighbor_offsets
    sparse = wid.sparse

    file = open("graph_"+str(distance)+".pkl", 'wb')
    r = pkl.dump(dict, file)
    file.close()

    print(distance)
