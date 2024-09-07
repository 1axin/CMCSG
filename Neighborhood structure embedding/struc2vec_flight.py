import csv

import numpy as np
import pandas as pd


from ge.classify import read_node_label,Classifier

from ge import Struc2Vec

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

import networkx as nx

from sklearn.manifold import TSNE

def store_csv(data, file_name):
    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    return








def plot_embeddings(embeddings,):

    X, Y = read_node_label('',skip_head=True)



    emb_list = []

    for k in X:

        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)



    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)



    color_idx = {}

    for i in range(len(X)):

        color_idx.setdefault(Y[i][0], [])

        color_idx[Y[i][0]].append(i)



    for c, idx in color_idx.items():

        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    plt.legend()

    plt.show()

if __name__ == "__main__":



    G = nx.read_edgelist('', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])


    # nx.draw(G, node_size=10, font_size=10, font_color='blue', font_weight='bold' )
    # plt.show()

    model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
    model.train(embed_size = 64)
    embeddings = model.get_embeddings()
    print(embeddings)

    with open('embeddings.csv', 'w') as f :
        [f.write('{0},{1}\n'.format(key, value)) for key, value in embeddings.items()]

    store_csv(list(dict.values(embeddings)), '')
    # store_csv(list(dict(embeddings())), 'keys.csv')

    print(embeddings.keys())
    key = []
    key.append(embeddings.keys())
    store_csv(key, '')
    # store_csv(embeddings, 'embeddings.csv')
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)