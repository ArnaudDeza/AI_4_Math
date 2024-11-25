from src.rewards.utils import *


def conj_0_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: (Aouchiche, 2006). Let \( G \) be a connected graph on \( n \geq 3 \) vertices. Then
    R(G) + \alpha(G) \leq n - 1 + \sqrt{n - 1}.
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    r = randic_index(G)
    alpha = len(nx.maximal_independent_set(G))
    info['randic_index'] = r
    info['alpha'] = alpha
    score = r + alpha - n + 1 - sqrt(n - 1)
    info['score'] = score
    return score, info

def conj_1_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 9
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    max_spectrum = max(nx.adjacency_spectrum(G).real)
    alpha = len(nx.maximal_independent_set(G))
    info['max_spectrum'] = max_spectrum
    info['alpha'] = alpha
    score = sqrt(n - 1) - n + 1 - max_spectrum + alpha
    info['score'] = score
    return score, info

def conj_2_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 8
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    
    proximity_val = proximity(G)
    connectivity_val = connectivity(G)
    info['proximity'] = proximity_val
    info['connectivity'] = connectivity_val
    if n == 0: score = 0.0
    elif n % 2 == 0: score = 0.5 * (n ** 2 / (n - 1)) * (1 - cos(pi / n))
    else: score = 0.5 * (n + 1) * (1 - cos(pi / n))

    score -= connectivity_val * proximity_val
    info['score'] = score
    return score, info
 

def conj_3_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 7
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    
    proximity_val = proximity(G) 
    info['proximity'] = proximity_val 
    max_spectrum = max(nx.adjacency_spectrum(G).real)
    info['max_spectrum'] = max_spectrum

    score = max_spectrum * proximity_val - n + 1
    info['score'] = score
    return score, info

def conj_4_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 6
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    gamma = len(nx.dominating_set(G))
    info['gamma'] = gamma
    if (2 * n - 2 * gamma) == 0:
        score = 0.0
    else:
        score = (1 - gamma) / (2 * n - 2 * gamma) + (gamma + 1) / 2 - mod_zagreb_2(G)
    info['score'] = score
    return score, info


def conj_5_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 5
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    mod_zagreb_val = mod_zagreb_2(G)
    info['mod_zagreb_val'] = mod_zagreb_val
    score = mod_zagreb_val - (n + 1) / 4
    info['score'] = score
    return score, info





def conj_6_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 4
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    spectrum = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    if len(spectrum) < 2:
        second_largest = 0.0
    else:
        second_largest = spectrum[1]

    harmonic_val = harmonic_index(G)
    
    info['harmonic_val'] = harmonic_val
    info['second_largest'] = second_largest
    score = second_largest - harmonic_val
    info['score'] = score
    return score, info




def conj_7_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 3
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    p_A_val, m_val, p_D_val = p_A(G), m(G),  p_D(G)
    info['p_A'] = p_A_val
    info['p_D'] = p_D_val
    info['m'] = m_val
    if m_val == 0 or n == 0:
        score = 0.0
    else:
        score = abs(p_A_val / m_val - (1 - p_D_val / n)) - 0.28
    info['score'] = score
    return score, info

def conj_8_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 2
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    proximity_val = proximity(G)
    diameter_val = nx.diameter(G)
    floor_val = floor(2 * diameter_val / 3)
    eigenval = dist_eigenvalue(G, floor_val)

    info['proximity'] = proximity_val
    info['diameter'] = diameter_val
    info['eigenval'] = eigenval

    score = -proximity_val - eigenval
    info['score'] = score
    return score, info

def conj_9_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 1
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    max_spectrum = max(nx.adjacency_spectrum(G).real)
    matching_size = len(nx.maximal_matching(G))

    info['max_spectrum'] = max_spectrum
    info['matching_size'] = matching_size

    score = sqrt(n - 1) + 1 - max_spectrum - matching_size
    info['score'] = score
    return score, info
    


































def conj_10_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    info[''] = ...
    info[''] = ...
    score = ...
    info['score'] = score
    return score, info




def conj_11_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    info[''] = ...
    info[''] = ...
    score = ...
    info['score'] = score
    return score, info



def conj_12_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    info[''] = ...
    info[''] = ...
    score = ...
    info['score'] = score
    return score, info


def conj_13_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    info[''] = ...
    info[''] = ...
    score = ...
    info['score'] = score
    return score, info



def conj_14_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    info[''] = ...
    info[''] = ...
    score = ...
    info['score'] = score
    return score, info


def conj_15_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture XXXXX
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    info[''] = ...
    info[''] = ...
    score = ...
    info['score'] = score
    return score, info