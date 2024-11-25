from src.rewards.utils import *






def conj0_score(input,INF):
    '''Score function for Conjecture 10
    (Aouchiche, 2006). Let \( G \) be a connected graph on \( n \geq 3 \) vertices. Then
    \[
    R(G) + \alpha(G) \leq n - 1 + \sqrt{n - 1}.
    \]
    '''
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    G = ...

    n = G.number_of_nodes()
    score = randic_index(G) + len(nx.maximal_independent_set(G)) - n + 1 - sqrt(n - 1)

    #TODO
    return score,info

def conj1_score(input,INF):
    '''Score function for Conjecture 9'''
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    G = ...
    n = G.number_of_nodes()
    if not nx.is_connected(G):
        return -1000
    score =  sqrt(n - 1) - n + 1 - max(eigvals(nx.adjacency_matrix(G).todense())) + len(nx.maximal_independent_set(G))
    return score,info

def conj2_score(input,INF):
    '''Score function for Conjecture 8'''
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    G = ...
    n = G.number_of_nodes()
    if n % 2 == 0:
        return 0.5 * (n ** 2 / (n - 1)) * (1 - cos(pi / n)) - connectivity(G) * proximity(G)
    else:
        return 0.5 * (n + 1) * (1 - cos(pi / n)) - connectivity(G) * proximity(G)

    return score,info

def conj3_score(input,INF):
    '''Score function for Conjecture 7'''
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    G = ...
    n = G.number_of_nodes()
    score =  max(eigvals(nx.adjacency_matrix(G).todense())) * proximity(G) - n + 1
    return score,info








def conj4_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj5_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj6_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj7_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj8_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj9_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj10_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj11_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj12_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj13_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj14_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj15_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj16_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj17_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj18_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def con19_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info

def conj20_score(input,INF):
    adjMatG, edgeListG, Gdeg = input
    info,score = {}, 0
    #TODO
    return score,info