import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from sklearn.cluster import KMeans
import sklearn

filename = 'ca-GrQc.txt'
edges = []
def adjacency_matrix(filename):
    f = open(filename)
    lines = f.readlines()
    print(lines[0])
    #number of vertices
    n = int(lines[0].split()[2])
    A = np.zeros((n,n))
    print(n)
    for line in lines[1:len(lines)]:
        node1, node2 = line.split()
        node1, node2 = int(node1), int(node2)
        edges.append([node1, node2])
        A[node1, node2] = 1
        A[node2, node1] = 1
    return(A, edges)

def degree_matrix(A):
    n = A.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        D[i][i] = sum(A[i])
    return D


def unorm_laplacian(A, D):
    n = A.shape[0]
    L = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            L[i][j] = D[i][j] - A[i][j]
    return(L)

def norm_laplacian(A, D):
    n = A.shape[0]
    I = np.identity(n)
    L = np.zeros((n,n))
    D_ = scipy.linalg.fractional_matrix_power(D, -1/2)
    L = I - np.dot(D_, np.dot(A,D_))
    return(L)

def mat_eigen(L, k):
    eigenvals, eigenvects = scipy.sparse.linalg.eigs(L, k)
    print('Eigenvalues : {}'.format(eigenvals))
    print('vectors : {}'.format(eigenvects))
    return(np.real(eigenvects))

def mat_eigen_norm(L, k):
    eigenvals, eigenvects = scipy.sparse.linalg.eigs(L, k)
    print('Eigenvalues : {}'.format(eigenvals))
    print('vectors : {}'.format(eigenvects))
    U = np.real(eigenvects)
    U = sklearn.preprocessing.normalize(U)
    return(U)


def kmeans(U, k):
    km = KMeans(n_clusters = k).fit(U)
    print('labels : {}'.format(km.labels_))
    predict = km.predict(U)
    return predict

#returns the edges with cluster ID
def edges_to_clusters(edges, pred):
    new_edges = []
    for i in range(len(edges)):
        new_edges.append([pred[edges[i][0]], pred[edges[i][1]]])
    #print('new {}'.format(new_edges))
    return new_edges

def phi(edges, pred, k):
    n = len(edges)
    E = 0
    sizes = [0]*k
    unique, counts = np.unique(pred, return_counts = True)
    minimum = min(counts)
    for i in range(n):
        if edges[i][0] != edges[i][1]:
            E+=1
    print('E : {}'.format(E))
    print('sizes : {}'.format(minimum))
    return(E/minimum)

def main(filename):
    A, edges = adjacency_matrix(filename)
    #print('shape : {}'.format(A.shape))
    #print('adjacency {}'.format(A))
    #print('taille : {}'.format(A.shape[0]))
    D = degree_matrix(A)
    #print('D : {}'.format(D))
    #L = unorm_laplacian(A,D)
    L = norm_laplacian(A,D)
    print('L : {}'.format(L))
    U = mat_eigen(L, 2)
    #print('U : {}'.format(U))
    pred  = kmeans(U, 2)
    #print('kmeans : {}'.format(km))
    #print(len(km))
    new_edges = edges_to_clusters(edges, pred)
    goodness = phi(new_edges, pred, 2)
    print('phi : {}'.format(goodness))
    return(True)

main(filename)
