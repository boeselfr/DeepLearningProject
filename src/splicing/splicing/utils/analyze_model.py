import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def analyze_graph_importance(model, modelname):

    for ix,target in enumerate(['None', 'Acceptor', 'Donor']):
        layer_weights = model['model']['out.weight'].data[ix, :, :]

        spliceai_features = layer_weights.size()[0] - model['settings'].hidden_size
        graph_features = model['settings'].hidden_size

        layer_weights_abs = torch.abs(layer_weights)
        graph_importance = torch.linalg.vector_norm(layer_weights_abs[-graph_features:,:], ord=2)
        spliceai_importance = torch.linalg.vector_norm(layer_weights_abs[:spliceai_features,:], ord=2)
        #summed_weigths = torch.sum(layer_weights_abs, axis=0)
        imp = [float(spliceai_importance),float(graph_importance)]
        importance = pd.DataFrame({'conv_weights':imp, 'information_source':['spliceai_features', 'graph features']})
        fig,ax= plt.subplots(2,1)
        g1 = sns.barplot(ax=ax[0], data=layer_weights_abs[:,0],
                    palette=['b' if i < (spliceai_features) else 'r' for i in range(spliceai_features + graph_features)])
        g1.set(xticklabels=[])
        g1.vlines(x=32, ymin=0, ymax=torch.max(layer_weights_abs[:,0]),color='black', linewidth=2)
        g1.set_title(f'normalized weights for: {target} - model: {modelname}', fontsize=5)
        sns.barplot(ax=ax[1],x='information_source', y='conv_weights', data=importance ,palette=['b', 'r'])
        fig.show()
        print(f'normalized graph importance: {graph_importance}')
        print(f'normalized spliceai importance: {spliceai_importance}')


def analyze_node_representation_to_graph_representation(model):
    layer_weights = model['model']['lin.weight'].data
    layer_weights_abs = torch.abs(layer_weights)
    summed_weigths = torch.sum(layer_weights_abs, axis=0)
    bar = sns.barplot(data=summed_weigths)
    plt.show()
    ax = sns.heatmap(layer_weights)
    plt.show()



if __name__ == '__main__':
    """parser = argparse.ArgumentParser(description='Define what to analyze.')
    parser.add_argument('--modelpath','-mpath', type=str, dest='modelpath',
                        default='/Users/fredericboesel/Documents/master/herbst21/deeplearning/data/results/SpliceAI_full_model_e1_cl400_g1.h5',
                        help='path to the model to analyze')
    parser.add_argument('--analyze_weigths_final_layer', '-fl_weigths', action='store_true',
                        dest='analyze_weigths_final_layer',
                        help='set this if you want to anaylze the final layer weights of the model')

    args = parser.parse_args()"""
    base_dir = '/Users/fredericboesel/Documents/master/herbst21/deeplearning/data/results/new'
    for modelname in os.listdir(base_dir):
        print('#############')
        print(modelname)

        try:
            model_path = os.path.join(base_dir, modelname, 'SpliceAI_e1_cl400_g1.h5')
            model = torch.load(model_path, map_location=torch.device('cpu'))
        except Exception as e:
            model_path = os.path.join(base_dir, modelname, 'SpliceAI_e10_cl400_g1.h5')
            model = torch.load(model_path, map_location=torch.device('cpu'))
        analyze_graph_importance(model, modelname)
