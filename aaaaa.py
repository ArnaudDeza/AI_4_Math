import numpy as np
from numba import njit
import math

import networkx as nx
import igraph as ig
@njit
def bfs(Gdeg,edgeListG):
    #simple breadth first search algorithm, from each vertex
    N = Gdeg.size
    
    distMat1 = np.zeros((N,N))
    conn = True
    for s in range(N):
        visited = np.zeros(N,dtype=np.int8)
     
        # Create a queue for BFS. Queues are not suported with njit yet so do it manually
        myQueue = np.zeros(N,dtype=np.int8)
        dist = np.zeros(N,dtype=np.int8)
        startInd = 0
        endInd = 0

        # Mark the source node as visited and enqueue it 
        myQueue[endInd] = s
        endInd += 1
        visited[s] = 1

        while endInd > startInd:
            pivot = myQueue[startInd]
            startInd += 1
            
            for i in range(Gdeg[pivot]):
                if visited[edgeListG[pivot][i]] == 0:
                    myQueue[endInd] = edgeListG[pivot][i]
                    dist[edgeListG[pivot][i]] = dist[pivot] + 1
                    endInd += 1
                    visited[edgeListG[pivot][i]] = 1
        if endInd < N:
            conn = False #not connected
        
        for i in range(N):
            distMat1[s][i] = dist[i]
        
    return distMat1, conn



def score_graph(adjMatG, edgeListG, Gdeg,pattern_graphs,use_nx_):

    


    # WRONG CODE
    num_induced = 0

    if use_nx_:
        G = nx.convert_matrix.from_numpy_array(adjMatG)
        for pattern_graph in pattern_graphs:
            GM = nx.isomorphism.GraphMatcher(G,pattern_graph)
            if GM.subgraph_is_isomorphic():
                num_induced += 1

        
    else:
        G = ig.Graph.from_networkx(nx.convert_matrix.from_numpy_array(adjMatG))
        for pattern_graph in pattern_graphs:
            
            g1 = ig.Graph.from_networkx(pattern_graph)
            if G.subisomorphic_lad(g1):
            #if G.subisomorphic_vf2(g1):
                num_induced += 1
        
    

    return num_induced
        


import networkx as nx



def load_pattern_graphs(k):
    if k == 2:            path = 'simple_graphs_mckay/graph2.g6'
    elif k == 3:          path = 'simple_graphs_mckay/graph3.g6'
    elif k == 4:          path = 'simple_graphs_mckay/graph4.g6'
    elif k == 5:          path = 'simple_graphs_mckay/graph5.g6'
    elif k == 6:          path = 'simple_graphs_mckay/graph6.g6'
    elif k == 7:          path = 'simple_graphs_mckay/graph7.g6'
    elif k == 8:          path = 'simple_graphs_mckay/graph8.g6'

    pattern_graphs = nx.read_graph6(path)
    return pattern_graphs








max_num_sols = 16
sols_found = 0
#N = number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
print('\n \n \t \t N = {}\n \n'.format(N))
# Load in pattern graphs
if N == 1:  
    k = 1
    max_score_attainable = 1
if N == 3:                           #  trivial
    k = 2
    max_score_attainable = 2           
if N == 5:                           #  trivial
    k = 3
    max_score_attainable = 4
if N == 8:                           #  trivial
    k = 4
    max_score_attainable = 11
if N == 10: 
    k = 5
    max_score_attainable = 34
if N == 14:  
    k = 6
    max_score_attainable = 156
if N == 16:  
    k = 7
    max_score_attainable = 1044
if N == 17:  
    k = 7
    max_score_attainable = 1044
if N == 18:  
    k = 7
    max_score_attainable = 1044

if os.path.exists(parent_folder)== False: os.mkdir(parent_folder)
save_folder = parent_folder  + '/search_for_n_{}__k_{}_opt_{}'.format(N, k, max_score_attainable)
if os.path.exists(save_folder)== True:shutil.rmtree(save_folder)
if os.path.exists(save_folder)== False: os.mkdir(save_folder)



pattern_graphs = load_pattern_graphs(k)
