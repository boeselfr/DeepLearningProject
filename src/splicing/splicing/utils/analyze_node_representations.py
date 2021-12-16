import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

def analyze_node_reps(node_rep_dict):
    for key in node_rep_dict.keys():
        print('########')
        print(key)
        std = torch.std(node_rep_dict[key], dim=0)
        normed_std = torch.linalg.vector_norm(std, ord=2)
        print(f'std_vector: {std}')
        print(f'normed_std: {normed_std}')
        #ts = torch.linalg.vector_norm(node_rep_dict[key], dim = 1).detach().numpy()
        plt.plot(node_rep_dict[key])
        plt.title(f'{key}')
        plt.show()
    return



if __name__ == '__main__':
    dict_path = '/Users/fredericboesel/Documents/master/herbst21/deeplearning/data/node_representations/node_representation_dict_chr4.pt'
    node_rep_dict = torch.load(dict_path)
    analyze_node_reps(node_rep_dict)