import numpy as np
from sklearn.cluster import SpectralClustering


def adjacency_matrix(filename):
    edges = []
    number_nodes = 0
    with open(filename, "r") as data:
        lines = data.readlines()
        for i in range (1, len(lines)):
            node1, node2 = lines[i].strip().split()
            node1 = int(node1)
            node2 = int(node2)
            edges.append([node1, node2])
            number_nodes = max(number_nodes, node1, node2)

    number_nodes += 1  # We only had the maximum index of the nodes
    # Initiate empty matrix
    adjacency_matrix = np.zeros((number_nodes, number_nodes))
    n = len(edges)
    print(edges[0][0])
    for node1, node2 in edges:
        adjacency_matrix[node1][node2] = 1
        adjacency_matrix[node2][node1] = 1
    return(adjacency_matrix)

def main(filename):
    adj_mat = adjacency_matrix(filename)
    sc = SpectralClustering(2, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    return(sc.labels)

main('ca-AstroPh.txt')
