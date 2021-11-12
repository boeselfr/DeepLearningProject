# function that given its location(chr and pos) returns a list of all adjacent windows
import os
import pickle
import numpy as np


def adjacency_lookup(chr, pos, adjacency_matrix, idx_dict):
    # round to next thousand for window lookup, (could alse be done to lower thousand to align with window)
    ####this currently leads to errors when using 5k grpahs as 1k rounding could end up with non dounf indexes!
    pos_rounded = round(pos, -3)
    try:
        idx = idx_dict[chr][pos_rounded]['bin_idx']
    except Exception as e:
        print(f'chr: {chr}, position: {pos} is not a valid location!')
        return [], []
    # lookup non zeros in matrix:
    # adjacency matrix[chr] is a sparse matrix that gets slcied and then converted to an array.
    # All entries that are non zero are returned in a list
    # at the moment the return is a list of indices, we can get the positions inside the chromosomes aswell:

    adjacencies_indices = np.where(adjacency_matrix[chr][idx].toarray()[0] > 0)[0].tolist()
    adjacencies_locations = list()

    for key, value in idx_dict[chr].items():
        if value['bin_idx'] in adjacencies_indices:
            adjacencies_locations.append(key)

    return adjacencies_indices, adjacencies_locations



if __name__ == '__main__':
    matrix_folder = '/Users/fredericboesel/Documents/master/herbst21/deeplearning/data/processed_data_allw/GM12878/5000/hic/'
    matrix_all = 'train_graphs_500000_SQRTVCnorm.pkl'
    idx_dict = 'test_vail_train_bin_dict_500000_SQRTVCnorm.pkl'


    with open(os.path.join(matrix_folder, matrix_all), "rb") as f:
        data = pickle.load(f)
        #print(data)

    with open(os.path.join(matrix_folder, idx_dict), "rb") as f:
        idx_data = pickle.load(f)
        #print(idx_data)

    chr = 'chr2'
    pos = 35000

    adjacencies_inidices, adjacencies_locations = adjacency_lookup(chr, pos, data, idx_data)
    print(f'indexes of 3d adjacent windows : {adjacencies_inidices} ')
    print(f'start positions of windows (on same chromosome as query): {adjacencies_locations}')


