import numpy as np
import networkx as nx


# Imports
from src.rl_algo.cem.utils import state_to_graph
from src.rewards.conjectures_graph_undirected import *


# Set the reward function based on the argument
reward_dict = {
    1 : conj_1_score,        # Conjecture 1 -- known counterexample with 9 nodes
    2 : conj_2_score,        # Conjecture 2 -- known counterexample with 8 nodes
    3 : conj_3_score,        # Conjecture 3 -- known counterexample with XXX nodes
    4 : conj_4_score,        # Conjecture 4 -- known counterexample with XXX nodes
    5 : conj_5_score,        # Conjecture 5 -- known counterexample with XXX nodes
    6 : conj_6_score,        # Conjecture 6 -- known counterexample with XXX nodes
    7 : conj_7_score,        # Conjecture 7 -- known counterexample with 7 nodes
    8 : conj_8_score,        # Conjecture 8 -- known counterexample with 6 nodes
    9 : conj_9_score,        # Conjecture 9 -- known counterexample with XXX nodes
    10 : conj_10_score,      # Conjecture 10 -- known counterexample with XXX nodes
    #11 : conj_11_score,      # Conjecture 11 -- known counterexample with XXX nodes
    #12 : conj_12_score,      # Conjecture 12 -- known counterexample with XXX nodes
    #13 : conj_13_score,      # Conjecture 13 -- known counterexample with XXX nodes
    #14 : conj_14_score,      # Conjecture 14 -- known counterexample with XXX nodes
    #15 : conj_15_score,      # Conjecture 15 -- known counterexample with XXX nodes
    #16 : conj_16_score,      # Conjecture 16 -- known counterexample with XXX nodes
    #17 : conj_17_score,      # Conjecture 17 -- known counterexample with XXX nodes
    #18 : conj_18_score,      # Conjecture 18 -- known counterexample with XXX nodes
    #19 : conj_19_score,      # Conjecture 19 -- known counterexample with XXX nodes
    #20 : conj_20_score,      # Conjecture 20 -- known counterexample with XXX nodes
    #21 : conj_21_score,      # Conjecture 21 -- known counterexample with XXX nodes
    }





def score_state_graph(args,state):
    # step 0: convert state to adj_matrix and G i.e networkx graph
    G, adjMatG, edgeListG, Gdeg = state_to_graph(args, state, directed = args.directed)
    
    # step 1: compute score
    score, information = args.calc_score(G, adjMatG, args.INF)

    # step 2: optional reward shaping:  If the graph is a tree, add a reward
    if args.reward__is_tree:
        if nx.is_tree(G):
            score += args.reward_weight_is_tree_pos
            information['is_tree'] = True
            information['cyclomatic_number'] = 0 

        else:
            score -= args.reward_weight_is_tree_neg
            information['is_tree'] = False
            # Step 2a: If we know that we do want a tree, look at tree properties that we may want:
            if args.reward__use_cylomatic:
                # Compute the cyclomatic number (number of cycles)
                num_components = nx.number_connected_components(G)
                cyclomatic_number = information['num_edges'] - information['num_nodes'] + num_components
                information['cyclomatic_number'] = cyclomatic_number
                # you want to minimize this as tree's have a cyclomatic number of 0
                score -= args.reward_weight_cyclomatic * cyclomatic_number
            else:
                information['cyclomatic_number'] = -1*args.INF

    # Update the information dictionary with final score used
    information['final_reward'] = score
    return score, information