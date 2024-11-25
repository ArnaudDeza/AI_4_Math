

import networkx as nx
import numpy as np
from numpy.linalg import eigvals
from numpy import sqrt, cos, pi, argmax, abs
from scipy.linalg import eig
from math import floor
import math
import scipy








  
def proximity(G):
    """
    Compute the proximity of a graph G.
    
    Parameters:
    - G: networkx.Graph - A NetworkX graph object
    
    Returns:
    - float: The proximity of the graph
    """
    n = len(G)
    shortest_paths = nx.floyd_warshall_numpy(G)  # Compute shortest paths as a dense matrix
    avg_distances = shortest_paths.sum(axis=1) / (n - 1)  # Average distance per node
    return avg_distances.min()  # Return the minimum average distance


def dist_eigenvalue(G, n):
    '''Returns the n-th largest eigenvalue of the distance matrix of G'''
    dist_matrix = nx.floyd_warshall_numpy(G)
    dist_spectrum = eigvals(dist_matrix)
    dist_spectrum.sort()
    return dist_spectrum[-n]

def mod_zagreb_2(G):
    '''Returns the modified second Zagreb index of G'''
    return sum(1 / (G.degree[u] * G.degree[v]) for u, v in G.edges)


def p_A(G):
    #Returns the peak location of the non-zero coefficients of the characteristic polynomial
    char_poly = np.poly(nx.adjacency_matrix(G).todense())
    coefs = np.abs(char_poly)
    nonzero_coefs = coefs[coefs != 0]
    return argmax(nonzero_coefs) + 1


def p_D(G):
    '''Returns the peak location of the normalized coefficients of the distance matrix'''
    dist_matrix = nx.floyd_warshall_numpy(G)
    char_poly = np.poly(dist_matrix)
    abs_coefs = np.abs(char_poly)
    n = G.number_of_nodes()
    norm_coefs = abs_coefs * [2**(k+2-n) for k in range(n + 1)]
    return argmax(norm_coefs)

def m(G):
    '''Returns the number of non-zero coefficients of CPA(G)'''
    char_poly = np.poly(nx.adjacency_matrix(G).todense())
    coefs = np.abs(char_poly)
    num_nonzero_coefs = np.sum(coefs != 0)
    return num_nonzero_coefs


def randic_index(graph):
    """ 
    Compute the Randic index of a graph efficiently.
    Parameters: graph (networkx.Graph): The input graph.
    Returns: float: The Randic index of the graph.
    """
    return sum(1 / math.sqrt(graph.degree[u] * graph.degree[v]) for u, v in graph.edges)


def harmonic_index(G):
    '''Returns the harmonic index of G'''
    return sum([2/(G.degree(u) + G.degree(v)) for u, v in G.edges()])


def connectivity(G):
    '''Returns the algebraic connectivity of G'''
    laplacian = nx.laplacian_matrix(G).todense()
    eigenvalues = np.sort(eigvals(laplacian))
    return eigenvalues[1]
