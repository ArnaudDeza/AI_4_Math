



from src.rewards.utils import *





def Conj1_score(G):                             # >>>>>>>>>>> GOOD works!!!!
    '''Score function for Conjecture 1'''
    n = G.number_of_nodes()
    if not nx.is_connected(G):
        return -1000
    return sqrt(n - 1) + 1 - max(eigvals(nx.adjacency_matrix(G).todense())) - len(nx.max_weight_matching(G))


def Conj2_score(G):
    '''Score function for Conjecture 2'''
    return -proximity(G) - dist_eigenvalue(G, floor(2 * nx.diameter(G) / 3))